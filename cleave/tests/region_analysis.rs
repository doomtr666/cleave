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
    let cps_program = convert_program(units, None);
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

/// `doc/plan-region-arena.md`'s own "Step 2" -- a real generalization of
/// this module, not the original shape. `shared_helper` is called *twice*
/// in the loop body, but each call site is individually, independently
/// non-escaping (`a`/`shared_helper(1)`'s own result are both only ever
/// combined via an ordinary arithmetic call, `Ring::add<i32>` -- itself a
/// *separate* real call, not a `PrimOp::Field` projection -- before the
/// *sum* alone reaches the loop's own carried state; neither operand is
/// itself in `escaping`, and neither has a `PrimOp::Field`-derived path to
/// it either). The *original* version of this analysis required exactly
/// one call site in the whole program and would have rejected this
/// unconditionally on the count alone -- needlessly, since both of
/// `shared_helper`'s own call sites are provably always inside the same
/// open region, exactly like a single safe call site would be. `region_
/// analysis::analyze`'s own relaxed condition ("every call site targeting
/// this callee is safe", not "there is only one") correctly marks it
/// region-local now. See the *next* test for the actual dangerous case
/// this relaxation must still reject: a *mix* of a safe and an unsafe call
/// site.
#[test]
fn a_function_called_twice_from_the_same_safe_loop_body_is_marked_local() {
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
        region_local.contains("shared_helper"),
        "shared_helper has two call sites, both inside the same loop body, \
         both individually non-escaping -- expected it region-local under \
         the relaxed \"every call site safe\" rule, got: {region_local:?}"
    );
}

/// The genuinely dangerous case the relaxation above must still reject: one
/// of `mixed_helper`'s own two call sites is safe (inside the loop, its own
/// result discarded into an escaping-unrelated local), the *other* directly
/// feeds the loop's own carried state (`acc`) -- marking `mixed_helper`
/// region-local wholesale would be unsound for *that* call site specifically
/// (its own internal allocations would need to survive past this loop
/// iteration's own region exit, since -- through the escaping call site --
/// they conceptually could). `region_specialize.rs`'s own job (a *separate*
/// module) is to split a callee like this into two names, one for each kind
/// of call site; `region_analysis::analyze` alone, with no specialization
/// pass run first, correctly leaves the untouched, still-shared name
/// excluded either way.
#[test]
fn a_function_with_one_safe_and_one_escaping_call_site_in_the_same_loop_is_never_marked_local() {
    let src = r#"
        fn mixed_helper(x: i32) -> i32 { x + 1 }

        fn main() -> i32 {
            let mut acc: i32 = 0;
            for _i in 0..10 {
                let _discarded = mixed_helper(1);
                acc = mixed_helper(acc);
            };
            acc
        }
        "#;
    let region_local = region_local_names(src);
    assert!(
        !region_local.contains("mixed_helper"),
        "mixed_helper's second call site directly feeds the loop's own \
         carried state -- must not be region-local even though its first \
         call site, alone, would qualify; got: {region_local:?}"
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

/// The real motivating shape for the transitive descent (`region_analysis.
/// rs`'s own extended doc comment on `find_region_local_functions`):
/// `helper_local` never computes anything itself, it delegates to `inner`
/// -- a plain, single-call-site helper reached *only* through `helper_
/// local`'s own body, never from anywhere else in the whole program.
/// `inner` should end up region-local too, not just `helper_local` itself.
#[test]
fn a_function_reached_only_through_an_already_region_local_callers_own_body_is_marked_local_too() {
    let src = r#"
        fn inner(x: i32) -> i32 { x * 2 }
        fn helper_local(x: i32) -> i32 { inner(x) + 1 }
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
        region_local.contains("inner"),
        "inner is reached only through helper_local's own body, itself already \
         confirmed region-local, with no other call site anywhere -- expected it \
         region-local too, got: {region_local:?}"
    );
}

/// **The dangerous case this whole extension exists to still get right**:
/// `shared` is called both from *inside* `helper_local` (an already-
/// region-local function) *and* from a completely unrelated place with no
/// region open at all (`main`'s own body, once, after the loop already
/// finished). Marking `shared` region-local would be sound for the call
/// *inside* `helper_local` alone, but would crash the very first allocation
/// at its *other* call site (`cleave_alloc_local` called with no region
/// open) -- `call_counts` is computed once, globally, specifically so this
/// case is caught at *any* recursion depth, not just at the top level.
#[test]
fn a_function_shared_between_a_region_local_callers_body_and_an_unrelated_call_site_is_never_marked_local() {
    let src = r#"
        fn shared(x: i32) -> i32 { x * 2 }
        fn helper_local(x: i32) -> i32 { shared(x) + 1 }
        fn helper_escaping(x: i32) -> i32 { x + 2 }

        fn main() -> i32 {
            let mut acc: i32 = 0;
            for _i in 0..10 {
                let a = helper_local(acc);
                acc = helper_escaping(a);
            };
            let extra = shared(acc);
            extra
        }
        "#;
    let region_local = region_local_names(src);
    assert!(
        !region_local.contains("shared"),
        "shared has a second, unrelated call site outside any loop at all -- \
         marking it region-local would crash that call site's own first \
         allocation with no region open; got: {region_local:?}"
    );
}

/// Multi-hop transitivity: `helper_local` (region-local) calls `mid`, which
/// itself calls `leaf` -- both `mid` and `leaf` have exactly one call site
/// in the whole program, two hops apart from the loop itself. The worklist
/// must keep descending, not stop after one level.
#[test]
fn transitive_descent_reaches_a_function_two_hops_deep() {
    let src = r#"
        fn leaf(x: i32) -> i32 { x + 1 }
        fn mid(x: i32) -> i32 { leaf(x) * 2 }
        fn helper_local(x: i32) -> i32 { mid(x) + 1 }
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
        region_local.contains("mid"),
        "mid is reached only through helper_local's own body -- expected region-local, got: {region_local:?}"
    );
    assert!(
        region_local.contains("leaf"),
        "leaf is reached only through mid's own body, two hops from the loop -- \
         expected the worklist to keep descending, got: {region_local:?}"
    );
}

/// A mutually-recursive pair reached through an already-region-local
/// caller -- `is_even`/`is_odd` each genuinely have *two* call sites in the
/// whole program (their own recursive partner, plus `helper_local`'s own
/// initial call into `is_even`), so `call_counts` correctly excludes both;
/// the real point of this test is that the worklist *terminates* rather
/// than looping forever bouncing between the two (it would, without the
/// `region_local.insert(...)` returning `false`-on-repeat guard) --
/// finishing at all, quickly, is the pass condition.
#[test]
fn a_mutually_recursive_pair_reached_through_a_region_local_caller_terminates_without_marking_either() {
    let src = r#"
        fn is_even(x: i32) -> bool {
            if x == 0 { true } else { is_odd(x - 1) }
        }
        fn is_odd(x: i32) -> bool {
            if x == 0 { false } else { is_even(x - 1) }
        }
        fn helper_local(x: i32) -> bool { is_even(x) }
        fn helper_escaping(x: bool) -> i32 { if x { 1 } else { 0 } }

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
        !region_local.contains("is_even") && !region_local.contains("is_odd"),
        "is_even/is_odd each have two call sites (their own mutual recursion, \
         plus helper_local's own initial call) -- neither individually satisfies \
         the single-call-site check, so neither should be region-local; got: {region_local:?}"
    );
}

/// The other genuinely-safe multi-site shape: `shared_across_loops` is
/// called from *two different* loops, in two different top-level functions
/// -- each occurrence individually non-escaping in its own loop. No
/// specialization is needed for this case at all: the relaxed "every call
/// site safe" rule already covers it directly, since `analyze`'s own
/// `safe_sites`/`sites_by_callee` are accumulated across the *whole*
/// program, not scoped to one loop at a time.
#[test]
fn a_function_called_safely_from_two_different_loops_is_marked_local() {
    let src = r#"
        fn shared_across_loops(x: i32) -> i32 { x + 1 }

        fn train(acc: i32) -> i32 {
            let mut a = acc;
            for _i in 0..10 {
                let r = shared_across_loops(a);
                a = r + 1;
            };
            a
        }

        fn evaluate_all(acc: i32) -> i32 {
            let mut a = acc;
            for _i in 0..5 {
                let r = shared_across_loops(a);
                a = r + 2;
            };
            a
        }

        fn main() -> i32 {
            evaluate_all(train(0))
        }
        "#;
    let region_local = region_local_names(src);
    assert!(
        region_local.contains("shared_across_loops"),
        "shared_across_loops has two call sites, in two different loops, \
         both individually non-escaping -- expected it region-local, got: {region_local:?}"
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
