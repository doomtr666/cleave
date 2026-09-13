//! Real tests for `cleave::region_specialize::specialize_region_local_
//! functions` — built against the actual CPS output of real cleave source
//! (`compile` + `collect_units` + `convert_program`), the same convention
//! `cleave/tests/region_analysis.rs` already establishes and explains.

use cleave::cps::{collect_mlir_types, collect_struct_schemas, collect_units, convert_program, CpsProgram};
use cleave::driver::compile;
use cleave::pipeline::check_type_errors;
use cleave::region_analysis::find_region_local_functions;
use cleave::region_specialize::specialize_region_local_functions;
use cleave::registry::Registry;

fn build_cps(src: &str) -> CpsProgram {
    let (result, _sources) = compile(vec![("test.cleave".to_string(), src.to_string())], &[]);
    let program = result.unwrap_or_else(|e| panic!("compile failed: {e:?}"));
    let registry = Registry::build(&program);
    if let Err(diags) = check_type_errors(&program, &registry) {
        panic!("type check failed: {diags:?}");
    }
    let units = collect_units(&program, &registry);
    let cps_program = convert_program(units, None);
    // Not read by either pass at all -- collected only because `collect_
    // units`'s own signature is shared with every other test file that
    // needs it, matching `region_analysis.rs`'s own test file for parity.
    let _ = collect_mlir_types(&program);
    let _ = collect_struct_schemas(&program);
    cps_program
}

fn names(program: &CpsProgram) -> std::collections::HashSet<String> {
    program.funcs.iter().map(|f| f.def.name.clone()).collect()
}

/// The real motivating shape (`doc/backlog.md`'s own "net_grad's own
/// remaining internal tensor allocations..." entry, `region_specialize.rs`'s
/// own module doc comment): `shared` is called from *inside* an already-
/// region-local caller's own body (safe) *and* from a completely unrelated
/// place with no region open at all (unsafe) -- exactly the case `region_
/// analysis::analyze` alone, correctly, still excludes (`cleave/tests/
/// region_analysis.rs`'s own `a_function_shared_between_a_region_local_
/// callers_body_and_an_unrelated_call_site_is_never_marked_local`).
/// Specialization must split it: a `shared$region` copy appears, `region_
/// analysis::find_region_local_functions` (run *after* specialization, the
/// real pipeline order) marks `shared$region` region-local, and the
/// original `shared` -- now serving only the unsafe call site -- stays
/// excluded.
#[test]
fn a_genuinely_mixed_callee_is_split_and_the_region_copy_is_marked_local() {
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
    let program = build_cps(src);
    let before = names(&program);
    assert!(
        !before.contains("shared$region"),
        "no split should exist before specialization runs, got: {before:?}"
    );

    let specialized = specialize_region_local_functions(program);
    let after = names(&specialized);
    assert!(
        after.contains("shared"),
        "the original name must still exist, serving the one unsafe call site, got: {after:?}"
    );
    assert!(
        after.contains("shared$region"),
        "a region-local copy must have been created for the one safe call site, got: {after:?}"
    );

    let region_local = find_region_local_functions(&specialized);
    assert!(
        region_local.contains("shared$region"),
        "shared$region's own only call site is the one already proven safe -- expected it \
         region-local, got: {region_local:?}"
    );
    assert!(
        !region_local.contains("shared"),
        "the original shared still serves the unrelated, unsafe call site -- must not be \
         region-local, got: {region_local:?}"
    );
}

/// A callee with exactly one call site, already safe, needs no split at
/// all -- `region_analysis::analyze`'s own relaxed rule already marks the
/// untouched original region-local directly (`doc/plan-region-arena.md`'s
/// own "Step 2" text: "a callee with zero eligible occurrences is
/// unaffected either way... one where *every* occurrence is already
/// eligible needs no split at all"). Checks `helper_local` specifically,
/// not the whole population: `x + 1`/`x + 2` both desugar to real calls
/// to the shared stdlib `Ring::add<i32>`, which genuinely *is* split here
/// (used inside `helper_local`'s own safe body *and* inside `helper_
/// escaping`'s own unsafe one) -- a real, correct positive from this same
/// pass, not a bug in it, just not what this specific test is about.
#[test]
fn a_single_safe_call_site_is_not_split() {
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
    let program = build_cps(src);
    let specialized = specialize_region_local_functions(program);
    let after = names(&specialized);
    assert!(
        !after.contains("helper_local$region"),
        "helper_local has a single, already-safe call site -- must not be split, got: {after:?}"
    );
    assert!(
        !after.contains("helper_escaping$region"),
        "helper_escaping has no safe call site at all -- must not be split, got: {after:?}"
    );
}

/// A callee called safely from two *different* loops needs no split either
/// (`cleave/tests/region_analysis.rs`'s own `a_function_called_safely_
/// from_two_different_loops_is_marked_local` already covers `analyze`
/// alone; this confirms specialization correctly leaves it untouched too,
/// rather than needlessly splitting a callee that's already wholesale
/// safe). Uses `*`/`+` deliberately kept apart from `train`/`evaluate_all`'s
/// own arithmetic (`+ 3`/`+ 4` there, `* 2` inside `shared_across_loops`
/// itself) so the underlying stdlib calls they desugar to don't
/// incidentally collide the way `a_single_safe_call_site_is_not_split`'s
/// own fixture does -- this test is specifically about `shared_across_
/// loops` itself, not about auditing every stdlib call transitively
/// reachable from it.
#[test]
fn a_callee_safe_at_every_call_site_across_two_loops_is_not_split() {
    let src = r#"
        fn shared_across_loops(x: i32) -> i32 { x * 2 }

        fn train(acc: i32) -> i32 {
            let mut a = acc;
            for _i in 0..10 {
                let r = shared_across_loops(a);
                a = r + 3;
            };
            a
        }

        fn evaluate_all(acc: i32) -> i32 {
            let mut a = acc;
            for _i in 0..5 {
                let r = shared_across_loops(a);
                a = r + 4;
            };
            a
        }

        fn main() -> i32 {
            evaluate_all(train(0))
        }
        "#;
    let program = build_cps(src);
    let specialized = specialize_region_local_functions(program);
    let after = names(&specialized);
    assert!(
        !after.contains("shared_across_loops$region"),
        "shared_across_loops is already wholesale safe at every one of its own call sites \
         -- must not be split, got: {after:?}"
    );
}

/// The clone's own body must be a real, independently-numbered copy, not
/// an alias of the original sharing the same `CVar`s -- `region_
/// specialize.rs`'s own `clone_and_renumber` doc comment has the full
/// reasoning for why a verbatim clone would be unsound (`op_lines`
/// corruption in particular). Checked structurally: every `CVar` reachable
/// from `shared$region`'s own params/body must be disjoint from every
/// `CVar` reachable from the *original* `shared`'s own params/body -- a
/// verbatim (bugged) clone would instead produce two identical sets.
#[test]
fn the_region_copys_own_cvars_are_disjoint_from_the_originals() {
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
    let program = build_cps(src);
    let specialized = specialize_region_local_functions(program);

    let original = specialized
        .funcs
        .iter()
        .find(|f| f.def.name == "shared")
        .expect("original shared must still exist");
    let region_copy = specialized
        .funcs
        .iter()
        .find(|f| f.def.name == "shared$region")
        .expect("shared$region must have been created");

    let original_vars = all_cvars_in(&original.def.body, &original.def.params);
    let region_vars = all_cvars_in(&region_copy.def.body, &region_copy.def.params);
    assert!(
        original_vars.is_disjoint(&region_vars),
        "the region copy's own CVars must never alias the original's -- \
         original: {original_vars:?}, region copy: {region_vars:?}"
    );
    // Sanity: both sets must be genuinely non-empty (params alone guarantee
    // this), so an accidentally-empty comparison can't pass vacuously.
    assert!(!original_vars.is_empty());
    assert!(!region_vars.is_empty());
}

fn all_cvars_in(expr: &cleave::cps::CExpr, params: &[cleave::cps::CVar]) -> std::collections::HashSet<cleave::cps::CVar> {
    use cleave::cps::{CExpr, CVal};
    let mut out: std::collections::HashSet<cleave::cps::CVar> = params.iter().copied().collect();
    fn walk(expr: &CExpr, out: &mut std::collections::HashSet<cleave::cps::CVar>) {
        match expr {
            CExpr::LetPrim { var, args, cont, .. } => {
                out.insert(*var);
                for a in args {
                    if let CVal::Var(v) = a {
                        out.insert(*v);
                    }
                }
                walk(cont, out);
            }
            CExpr::App { func, args } => {
                if let CVal::Var(v) = func {
                    out.insert(*v);
                }
                for a in args {
                    if let CVal::Var(v) = a {
                        out.insert(*v);
                    }
                }
            }
            CExpr::If {
                cond,
                then_branch,
                else_branch,
            } => {
                if let CVal::Var(v) = cond {
                    out.insert(*v);
                }
                walk(then_branch, out);
                walk(else_branch, out);
            }
            CExpr::Fix { defs, body } => {
                for d in defs {
                    out.extend(d.params.iter().copied());
                    walk(&d.body, out);
                }
                walk(body, out);
            }
        }
    }
    walk(expr, &mut out);
    out
}
