//! Real tests for `cleave::region_analysis::find_region_local_functions` —
//! built against the actual CPS output of real cleave source (`compile` +
//! `collect_units` + `convert_program`, the same pipeline stage every other
//! test file in this project uses), not hand-built `CpsProgram` values —
//! the analysis operates on the *exact* shapes `cps.rs`'s own conversion
//! produces (`lower_real_call`'s own documented `Fix{defs:[k], body:App{...
//! }}` shape in particular), which would be easy to get subtly wrong by
//! hand-constructing a "plausible-looking" CPS tree instead.

use cleave::cps::{collect_mlir_types, collect_struct_schemas, collect_units, convert_program};
use cleave::driver::compile;
use cleave::pipeline::check_type_errors;
use cleave::region_analysis::find_region_local_functions;
use cleave::registry::Registry;

fn region_local_names(src: &str) -> std::collections::HashSet<String> {
    let (result, _sources) = compile(vec![("test.cleave".to_string(), src.to_string())], &[]);
    let program = result.unwrap_or_else(|e| panic!("compile failed: {e:?}"));
    let registry = Registry::build(&program);
    if let Err(diags) = check_type_errors(&program, &registry) {
        panic!("type check failed: {diags:?}");
    }
    let units = collect_units(&program, &registry);
    let cps_program = convert_program(units);
    // Not read by this analysis at all -- collected only because `collect_
    // units`'s own signature is shared with every other test file that
    // needs it; kept here for parity, not because `find_region_local_
    // functions` itself uses either.
    let _ = collect_mlir_types(&program);
    let _ = collect_struct_schemas(&program);
    find_region_local_functions(&cps_program)
}

/// The exact shape `examples/mnist-interop`'s own training loop has:
/// `helper_local`'s own result is read once, by `helper_escaping`, and
/// never carried past this same iteration; `helper_escaping`'s own result
/// *becomes* the loop's own carried state. `helper_local` alone should be
/// marked region-local.
#[test]
fn a_call_whose_result_never_reaches_the_carried_state_is_marked_local() {
    let src = r#"
        fn helper_local(x: i32) -> i32 { x + 1 }
        fn helper_escaping(x: i32) -> i32 { x + 2 }

        fn main() -> i32 {
            let mut acc: i32 = 0;
            for _i in 0..10 {
                let a = helper_local(acc);
                acc = helper_escaping(a);
            };
            acc
        }
        "#;
    let region_local = region_local_names(src);
    assert!(
        region_local.contains("helper_local"),
        "helper_local's own result never reaches the carried state -- expected it region-local, got: {region_local:?}"
    );
    assert!(
        !region_local.contains("helper_escaping"),
        "helper_escaping's own result *becomes* `acc`, carried to the next iteration -- must not be region-local, got: {region_local:?}"
    );
}

/// `net_grad`/`Optimizer::step`'s own real shape: a helper's result reaches
/// the carried state only through a *field* projection (`g.2`, here `pair.
/// 1`), not directly -- the analysis must trace through `PrimOp::Field`,
/// not just check literal identity.
#[test]
fn a_result_reaching_the_carried_state_through_a_field_projection_is_not_local() {
    let src = r#"
        struct Pair { first: i32, second: i32 }

        fn make_pair(x: i32) -> Pair { Pair(first: x, second: x + 1) }

        fn main() -> i32 {
            let mut acc: i32 = 0;
            for _i in 0..10 {
                let pair = make_pair(acc);
                acc = pair.second;
            };
            acc
        }
        "#;
    let region_local = region_local_names(src);
    assert!(
        !region_local.contains("make_pair"),
        "make_pair's own result reaches the carried state through `.second` -- must not be region-local, got: {region_local:?}"
    );
}

/// The real bug found live in `examples/mnist-interop`'s own nested epoch/
/// batch loops (`Optimizer::step<Sgd, Network, ...>`, wrongly marked region-
/// local despite its own result becoming `net`/`state`, the *training*
/// loop's own carried state) -- a real, found-by-testing correctness bug in
/// `find_loops_and_mark`'s own nested-loop handling, not a hypothetical.
///
/// `helper_escaping`'s own call sits inside the *inner* loop, whose own
/// self-recursive tail-call is exactly where its result escapes to (`acc`,
/// read again next inner iteration) -- the *inner* loop's own dedicated
/// `analyze_loop_body` call gets this right on its own. The bug: `find_
/// loops_and_mark` also descends into the *outer* loop's own `then_branch`,
/// which textually contains the *entire* inner loop -- and `collect_
/// escaping`/`collect_calls_and_derivations`, run there relative to the
/// *outer* loop's own (different) self-recursive tail-call, never recognize
/// the *inner* loop's own tail-call as an escape at all (wrong loop name),
/// so `helper_escaping`'s result looks, from the outer loop's own
/// perspective, like it escapes nowhere -- a false "safe" verdict.
/// `find_region_local_functions`'s own whole-program `HashSet` is a union
/// across every loop's own analysis, never an intersection, so this one
/// wrong verdict from the *outer* loop poisons the result even though the
/// *inner* loop's own analysis already got it right.
#[test]
fn a_call_escaping_only_via_an_inner_loops_own_carried_state_is_never_marked_local_even_when_an_outer_loop_wraps_it()
 {
    let src = r#"
        fn helper_escaping(x: i32) -> i32 { x + 2 }

        fn main() -> i32 {
            let mut acc: i32 = 0;
            for _epoch in 0..3 {
                for _batch in 0..10 {
                    acc = helper_escaping(acc);
                };
            };
            acc
        }
        "#;
    let region_local = region_local_names(src);
    assert!(
        !region_local.contains("helper_escaping"),
        "helper_escaping's own result becomes the *inner* loop's own carried state (`acc`), read again next inner iteration -- never safe to arena-allocate, regardless of the outer loop wrapping it; got: {region_local:?}"
    );
}

/// A function called from more than one place in the whole program is
/// never marked region-local, even if *this one* call site would otherwise
/// qualify -- `region_analysis.rs`'s own module doc comment has the real
/// reasoning (marking it local would be sound for *this* call site alone,
/// but not for whichever other call site doesn't have a region open at
/// all).
#[test]
fn a_function_called_from_more_than_one_place_is_never_marked_local() {
    let src = r#"
        fn shared_helper(x: i32) -> i32 { x + 1 }

        fn main() -> i32 {
            let mut acc: i32 = 0;
            for _i in 0..10 {
                let a = shared_helper(acc);
                acc = a + shared_helper(1);
            };
            acc
        }
        "#;
    let region_local = region_local_names(src);
    assert!(
        !region_local.contains("shared_helper"),
        "shared_helper has two call sites in the loop body alone -- must not be region-local, got: {region_local:?}"
    );
}

/// A real, found-by-testing bug (`doc/backlog-done.md`'s own "`region_
/// analysis.rs`'s escape analysis walked a loop's own exit branch, not
/// just its repeating body" entry, root-caused against `examples/
/// convex_hull.cleave --run`, which crashed a real `cleave_alloc_local`
/// call with no region open): a function called exactly *once*, structurally
/// *after* an earlier loop finishes (never inside any loop's own repeating
/// body at all) must never be marked region-local. `mlir_lower.rs::
/// lower_loop`'s own doc comment is explicit that a loop's `else_branch`
/// ("loop exit") "runs in the *outer* scope, ordinary flow" -- lowered
/// entirely outside the `scf.while` op, with no `cleave_region_enter`/
/// `cleave_region_exit` pair around it at all. The old, broken version of
/// `analyze_loop_body` scanned the *whole* `loop_def.body` (condition chain
/// + `then_branch` + `else_branch` together) for calls -- since a CPS-
/// converted loop's own exit path structurally *contains* the rest of the
/// enclosing function as part of the same term, `before_body`'s call here
/// (found only in the `for` loop's own `else_branch`) was wrongly swept in
/// as if it ran once per iteration.
#[test]
fn a_call_after_an_earlier_loop_finishes_is_never_marked_local() {
    let src = r#"
        fn before_body(x: i32) -> i32 { x + 100 }
        fn loop_helper(x: i32) -> i32 { x + 1 }

        fn main() -> i32 {
            let mut acc: i32 = 0;
            for _i in 0..3 {
                acc = loop_helper(acc);
            };
            let b = before_body(acc);
            b
        }
        "#;
    let region_local = region_local_names(src);
    assert!(
        !region_local.contains("before_body"),
        "before_body is called exactly once, only after the loop has already \
         finished -- its own call site is never wrapped in a `cleave_region_enter`/\
         `cleave_region_exit` pair, so marking it region-local crashes the very \
         first allocation inside it; got: {region_local:?}"
    );
}

/// A program with no loop at all -- the analysis must find nothing to mark,
/// not panic or misfire on the "no `Fix` is ever self-recursive" case.
#[test]
fn a_program_with_no_loop_marks_nothing_region_local() {
    let src = r#"
        fn helper(x: i32) -> i32 { x + 1 }
        fn main() -> i32 { helper(41) }
        "#;
    let region_local = region_local_names(src);
    assert!(
        region_local.is_empty(),
        "no loop exists in this program -- nothing should be marked region-local, got: {region_local:?}"
    );
}
