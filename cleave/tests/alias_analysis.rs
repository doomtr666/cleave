//! Tests for `cleave::alias_analysis` — see that module's own doc comment
//! for the full design. Every test compiles through the exact pipeline
//! stage this analysis is meant to run at: e-graph-optimized CPS, *before*
//! `refcount::insert_refcounting` ever runs (no `Retain`/`Release` exist
//! yet at this point — the whole reason the analysis is self-contained
//! rather than reading those markers, `alias_analysis.rs`'s own doc
//! comment explains why).
//!
//! Several tests use an `extern fn` sink with no real implementation
//! purely to stop the e-graph from fusing a small function's whole body
//! away (confirmed necessary directly: a naive, unprotected version of
//! several of these collapsed to a single constant before this analysis
//! ever saw a real call at all — matching the exact trap `cleave/tests/
//! refcount.rs`'s own regression tests for the sibling `identity_param_
//! positions` fix had to route around the same way).

use cleave::alias_analysis::{analyze, analyze_identity};
use cleave::driver::compile;
use cleave::egraph::optimize_program;
use cleave::pipeline::check_type_errors;
use cleave::registry::Registry;

/// The exact CPS shape `alias_analysis::analyze` is meant to run on --
/// e-graph-optimized, dead-code-eliminated, but *before* any refcounting
/// pass has touched it. Mirrors `pipeline.rs::build_optimized_cps`'s own
/// prefix, stopping short of `insert_refcounting`.
fn optimized_cps(src: &str) -> cleave::cps::CpsProgram {
    let (result, _sources) = compile(vec![("test.cleave".to_string(), src.to_string())], &[]);
    let program = result.unwrap_or_else(|e| panic!("compile failed: {e:?}"));
    let registry = Registry::build(&program);
    if let Err(diags) = check_type_errors(&program, &registry) {
        panic!("type check failed: {diags:?}");
    }
    let units = cleave::cps::collect_units(&program, &registry);
    let cps_program = cleave::cps::convert_program(units, None);
    let cps_program = cleave::cps::eliminate_dead_code(cps_program);
    let (cps_program, _) = optimize_program(cps_program, &registry, false);
    cleave::cps::eliminate_dead_code(cps_program)
}

/// The plainest possible never-aliased chain: `bump` only ever borrows its
/// own parameter (reads two fields off it) and returns a completely fresh
/// struct — never embeds `a` itself anywhere, never returns it unchanged.
/// `bump`'s own parameter 0 must not be marked aliased.
#[test]
fn a_pure_borrow_and_fresh_construction_is_never_aliased() {
    let src = "
        struct Boxed { v: i32, tag: [i32; 1] }
        fn bump(a: Boxed) -> Boxed { Boxed(v: a.v + 1, tag: [a.tag[0]]) }
        extern fn opaque_sink(b: Boxed) -> i32;
        extern fn opaque_sink2(b: Boxed) -> i32;
        fn main() -> i32 {
            let a = Boxed(v: 1, tag: [0]);
            opaque_sink(bump(a)) + opaque_sink2(a)
        }
        ";
    let program = optimized_cps(src);
    let summary = analyze(&program);
    assert!(
        !summary.is_aliased("bump", 0),
        "`bump` only ever borrows `a` and returns a fresh struct -- its own \
         parameter must not be marked aliased"
    );
}

/// The same parameter embedded into *two different fields* of one fresh
/// struct (`Struct(a: p, b: p)`) — two independent references to the one
/// allocation from a single construction, no later use required at all.
#[test]
fn a_parameter_embedded_twice_in_the_same_construction_is_aliased() {
    let src = "
        struct Boxed { v: i32, tag: [i32; 1] }
        struct Pair { first: Boxed, second: Boxed }
        fn duplicate(p: Boxed) -> Pair { Pair(first: p, second: p) }
        extern fn opaque_sink(x: Pair) -> i32;
        fn main() -> i32 {
            let a = Boxed(v: 1, tag: [0]);
            opaque_sink(duplicate(a))
        }
        ";
    let program = optimized_cps(src);
    let summary = analyze(&program);
    assert!(
        summary.is_aliased("duplicate", 0),
        "`duplicate` embeds `p` into two fields of the same fresh `Pair` -- \
         its own parameter must be marked aliased"
    );
}

/// Rule 4b, the module's own doc comment has the full reasoning: a
/// parameter embedded once into an *array* literal, and never referenced
/// again *by that same name*, must still be marked aliased -- unlike a
/// `Struct`'s own named fields, a later `Load` off the array can read any
/// originally-embedded element back out at a genuinely runtime index, so
/// rule 2's occurs-check (only ever looking for the original name) is
/// structurally blind to this. This is the exact, real gap behind
/// `examples/convex_hull.cleave --run`'s own non-deterministic use-after-
/// free (`doc/backlog.md`'s own entry has the full failure signature): each
/// of 8 fresh `Point`s was embedded once into a `[Point; 8]` array literal
/// and never referenced again by name, so the pre-fix analysis wrongly
/// classified every one of them as pool-eligible even though `points
/// [current]` (a later, runtime-indexed `Load`) is pushed into a
/// `DynArray<Point>` that clearly outlives the pool block backing it.
#[test]
fn a_parameter_embedded_once_into_an_array_literal_is_aliased_even_though_its_own_name_is_never_referenced_again() {
    let src = "
        struct Boxed { v: i32, tag: [i32; 1] }
        fn wrap(a: Boxed) -> [Boxed; 1] { [a] }
        extern fn opaque_array_sink(arr: [Boxed; 1]) -> i32;
        fn main() -> i32 {
            let a = Boxed(v: 1, tag: [0]);
            opaque_array_sink(wrap(a))
        }
        ";
    let program = optimized_cps(src);
    let summary = analyze(&program);
    assert!(
        summary.is_aliased("wrap", 0),
        "`wrap` embeds `a` into a `[Boxed; 1]` array literal and never \
         references it again by name -- its own parameter must still be \
         marked aliased, since a later `Load` off that array could read it \
         back out at a runtime index this analysis can't trace"
    );
}

/// `analyze_identity`'s own generalization -- see its doc comment for the
/// full motivation. A parameter threaded through an `if`/`else` where
/// *each* branch makes a real call before converging on a shared local
/// join point, only returning the parameter unchanged *after* that join,
/// must still be proven identity-shaped -- the exact shape `examples/
/// complex.cleave --run`'s own real, deterministic segfault traced back to
/// (`doc/backlog.md`'s own entry has the full story): neither the old rule
/// 1 (only a *literal* direct return) nor the old rule 2 (only a *single*
/// real-call resumption, never recursing further) could see through a
/// plain local join point at all.
#[test]
fn a_parameter_threaded_through_an_if_else_that_converges_via_a_join_before_returning_is_identity_shaped() {
    let src = "
        struct Acc { v: i32 }
        extern fn opaque_sink(x: i32) -> i32;
        fn touch1(a: Acc) -> i32 { a.v }
        fn touch2(a: Acc) -> i32 { a.v + 1 }
        fn thread_through(cond: bool, acc: Acc) -> Acc {
            if cond {
                opaque_sink(touch1(acc));
                acc
            } else {
                opaque_sink(touch2(acc));
                acc
            }
        }
        fn main() -> i32 {
            let a = Acc(v: 1);
            let a2 = thread_through(true, a);
            opaque_sink(a2.v)
        }
        ";
    let program = optimized_cps(src);
    let identity = analyze_identity(&program);
    assert_eq!(
        identity.returns_unchanged("thread_through", 1),
        Some(true),
        "`thread_through`'s own `acc` parameter is threaded through an if/else \
         where each branch makes a real call before converging on a shared \
         join point, then returns `acc` unchanged -- this must be proven \
         identity-shaped despite never directly returning `acc` and never \
         being a single-hop resumption forward"
    );
}

/// The parameter embedded once, then read again afterward — the classic
/// "protected embedding" hazard, this time detected directly (no
/// `refcount.rs`-inserted `Retain` exists yet at this pipeline stage).
#[test]
fn a_parameter_embedded_once_and_reused_afterward_is_aliased() {
    let src = "
        struct Boxed { v: i32, tag: [i32; 1] }
        struct Wrapper { inner: Boxed }
        extern fn opaque_sink(w: Wrapper) -> i32;
        extern fn opaque_sink2(b: Boxed) -> i32;
        fn wrap_and_reuse(p: Boxed) -> i32 {
            let w = Wrapper(inner: p);
            opaque_sink(w) + opaque_sink2(p)
        }
        fn main() -> i32 {
            let a = Boxed(v: 1, tag: [0]);
            wrap_and_reuse(a)
        }
        ";
    let program = optimized_cps(src);
    let summary = analyze(&program);
    assert!(
        summary.is_aliased("wrap_and_reuse", 0),
        "`p` is embedded into `w` and then read again via `opaque_sink2(p)` \
         -- its own parameter must be marked aliased"
    );
}

/// A genuinely identity-shaped function (returns its own parameter
/// unchanged, embeds it nowhere) must NOT be marked aliased on its own
/// side, per the module's own rule 5 -- the aliasing hazard for this shape
/// belongs entirely to the *caller*, handled separately by
/// `refcount::collect_identity_param_positions`.
#[test]
fn a_genuinely_identity_shaped_function_does_not_mark_its_own_parameter() {
    let src = "
        struct Boxed { v: i32, tag: [i32; 1] }
        fn id(x: Boxed) -> Boxed { x }
        extern fn opaque_sink(b: Boxed) -> i32;
        extern fn opaque_sink2(b: Boxed) -> i32;
        fn main() -> i32 {
            let a = Boxed(v: 1, tag: [0]);
            opaque_sink(id(a)) + opaque_sink2(a)
        }
        ";
    let program = optimized_cps(src);
    let summary = analyze(&program);
    assert!(
        !summary.is_aliased("id", 0),
        "`id` embeds its own parameter nowhere and returns it unchanged -- \
         it must not be marked aliased on the callee's own side"
    );
}

/// A parameter passed straight through to another function whose own
/// matching parameter position *is* aliased must propagate -- the
/// interprocedural half (rule 3), not just the local one.
#[test]
fn aliasing_propagates_through_a_direct_call() {
    let src = "
        struct Boxed { v: i32, tag: [i32; 1] }
        struct Pair { first: Boxed, second: Boxed }
        fn duplicate(p: Boxed) -> Pair { Pair(first: p, second: p) }
        fn forward(q: Boxed) -> Pair { duplicate(q) }
        extern fn opaque_sink(x: Pair) -> i32;
        fn main() -> i32 {
            let a = Boxed(v: 1, tag: [0]);
            opaque_sink(forward(a))
        }
        ";
    let program = optimized_cps(src);
    let summary = analyze(&program);
    assert!(
        summary.is_aliased("forward", 0),
        "`forward` passes `q` straight into `duplicate`, whose own matching \
         parameter is aliased -- `forward`'s own parameter must be marked \
         aliased too, by propagation"
    );
}

/// A parameter hand-carried to a function with no visible body at all (an
/// `extern fn`) has no evidence either way -- must default to aliased,
/// never the more permissive answer.
#[test]
fn a_parameter_passed_to_an_extern_fn_is_conservatively_aliased() {
    let src = "
        struct Boxed { v: i32, tag: [i32; 1] }
        extern fn opaque_sink(b: Boxed) -> i32;
        fn forward_to_extern(p: Boxed) -> i32 { opaque_sink(p) }
        fn main() -> i32 {
            let a = Boxed(v: 1, tag: [0]);
            forward_to_extern(a)
        }
        ";
    let program = optimized_cps(src);
    let summary = analyze(&program);
    assert!(
        summary.is_aliased("forward_to_extern", 0),
        "an extern fn's own body is never visible -- no evidence it doesn't \
         keep its own copy of `p`, so this must default to aliased"
    );
}

/// Mutual recursion between two top-level functions, each carrying a
/// refcounted parameter around the cycle without ever committing it
/// anywhere -- the whole point of doing this as a real fixed point rather
/// than a single-pass walk: neither function's own summary can be fully
/// known without the other's, and the answer here is `false` for both
/// (the value is only ever borrowed, all the way around the cycle).
#[test]
fn mutual_recursion_without_any_commitment_converges_to_not_aliased() {
    let src = "
        struct Boxed { v: i32, tag: [i32; 1] }
        fn ping(a: Boxed) -> i32 {
            if a.v > 0 { pong(a) } else { a.v }
        }
        fn pong(b: Boxed) -> i32 {
            if b.v > 0 { ping(Boxed(v: b.v - 1, tag: [0])) } else { b.v }
        }
        fn main() -> i32 {
            ping(Boxed(v: 3, tag: [0]))
        }
        ";
    let program = optimized_cps(src);
    let summary = analyze(&program);
    assert!(
        !summary.is_aliased("ping", 0),
        "`ping`'s own parameter is only ever borrowed (read, or passed on) \
         around the mutual recursion -- must not be marked aliased"
    );
    assert!(
        !summary.is_aliased("pong", 0),
        "same for `pong`'s own parameter, by symmetry"
    );
}

/// Mutual recursion where one side of the cycle *does* commit the value
/// (embeds it, still-referenced) -- both functions' own summaries must
/// come out aliased, since the cycle carries the same hazard around it
/// regardless of which specific call in the loop a caller happens to hit.
#[test]
fn mutual_recursion_with_a_commitment_on_one_side_marks_both() {
    let src = "
        struct Boxed { v: i32, tag: [i32; 1] }
        struct Pair { first: Boxed, second: Boxed }
        extern fn opaque_sink(x: Pair) -> i32;
        fn ping2(a: Boxed) -> i32 {
            if a.v > 0 { pong2(a) } else { 0 }
        }
        fn pong2(b: Boxed) -> i32 {
            if b.v > 0 {
                let p = Pair(first: b, second: b);
                opaque_sink(p) + ping2(Boxed(v: b.v - 1, tag: [0]))
            } else {
                0
            }
        }
        fn main() -> i32 {
            ping2(Boxed(v: 3, tag: [0]))
        }
        ";
    let program = optimized_cps(src);
    let summary = analyze(&program);
    assert!(
        summary.is_aliased("pong2", 0),
        "`pong2` embeds `b` twice into `p` directly -- must be aliased"
    );
    assert!(
        summary.is_aliased("ping2", 0),
        "`ping2` passes `a` straight into `pong2`'s own aliased position -- \
         must propagate back across the cycle"
    );
}

/// `value_is_ever_aliased` -- the per-VALUE query `doc/plan-affine-
/// ownership.md`'s Stage 2 actually needs at a construction site: not
/// "is this function's own parameter position ever aliased" but "does
/// *this one local binding*, from here to wherever it's last used, ever
/// get committed twice, reused after a commitment, or handed to an
/// already-known-aliased (or unknown) parameter position".
mod value_level {
    use super::*;
    use cleave::alias_analysis::{analyze, value_is_ever_aliased};
    use cleave::cps::CVar;

    /// Finds the `CVar` bound by the *first* `PrimOp::Struct` construction
    /// in `f_name`'s own body -- good enough for these tests, which each
    /// have exactly one construction of interest.
    fn first_struct_var(program: &cleave::cps::CpsProgram, f_name: &str) -> CVar {
        fn walk(expr: &cleave::cps::CExpr) -> Option<CVar> {
            use cleave::cps::{CExpr, PrimOp};
            match expr {
                CExpr::LetPrim { var, op, cont, .. } => {
                    if matches!(op, PrimOp::Struct(..)) {
                        Some(*var)
                    } else {
                        walk(cont)
                    }
                }
                CExpr::App { .. } => None,
                CExpr::If {
                    then_branch,
                    else_branch,
                    ..
                } => walk(then_branch).or_else(|| walk(else_branch)),
                CExpr::Fix { defs, body } => {
                    defs.iter().find_map(|d| walk(&d.body)).or_else(|| walk(body))
                }
            }
        }
        let f = program
            .funcs
            .iter()
            .find(|f| f.def.name == f_name)
            .unwrap_or_else(|| panic!("no function named `{f_name}`"));
        walk(&f.def.body).unwrap_or_else(|| panic!("no struct construction found in `{f_name}`"))
    }

    // A "constructed, then only read once, nothing else" test is
    // deliberately not included here: confirmed directly, on both a
    // single-shot and a loop-carried version, that this e-graph always
    // folds such a construction away entirely (`Boxed(v: x, ..).v`
    // reduces straight to `x`, looped or not) -- there is never a real
    // allocation left for this analysis to be asked about, which is the
    // correct, desirable optimization outcome, not a gap in this test
    // suite. `a_locally_constructed_value_passed_only_to_a_pure_borrow_
    // is_never_aliased` below is the realistic "never aliased, survives
    // as a real construction" case.

    /// The same local value, this time embedded into a struct and *also*
    /// read again afterward -- the value-level mirror of the parameter-
    /// level "embedded once and reused" test above.
    #[test]
    fn a_locally_constructed_value_embedded_and_reused_is_aliased() {
        let src = "
            struct Boxed { v: i32, tag: [i32; 1] }
            struct Wrapper { inner: Boxed }
            extern fn opaque_sink(w: Wrapper) -> i32;
            extern fn opaque_sink2(b: Boxed) -> i32;
            fn main() -> i32 {
                let a = Boxed(v: 1, tag: [0]);
                let w = Wrapper(inner: a);
                opaque_sink(w) + opaque_sink2(a)
            }
            ";
        let program = optimized_cps(src);
        let summary = analyze(&program);
        let var = first_struct_var(&program, "main");
        let f = program.funcs.iter().find(|f| f.def.name == "main").unwrap();
        assert!(
            value_is_ever_aliased(var, &f.def.body, &summary),
            "`a` is embedded into `w` and read again afterward -- must be \
             marked aliased"
        );
    }

    /// A locally-constructed value passed into a callee whose own matching
    /// parameter position `analyze` already proved aliased -- the O(1)
    /// lookup into the whole-program summary, not a re-walk of `duplicate`.
    #[test]
    fn a_locally_constructed_value_passed_to_an_aliasing_callee_is_aliased() {
        let src = "
            struct Boxed { v: i32, tag: [i32; 1] }
            struct Pair { first: Boxed, second: Boxed }
            fn duplicate(p: Boxed) -> Pair { Pair(first: p, second: p) }
            extern fn opaque_sink(x: Pair) -> i32;
            fn main() -> i32 {
                let a = Boxed(v: 1, tag: [0]);
                opaque_sink(duplicate(a))
            }
            ";
        let program = optimized_cps(src);
        let summary = analyze(&program);
        let var = first_struct_var(&program, "main");
        let f = program.funcs.iter().find(|f| f.def.name == "main").unwrap();
        assert!(
            value_is_ever_aliased(var, &f.def.body, &summary),
            "`a` is passed into `duplicate`, whose own parameter 0 is \
             already known aliased -- must be marked aliased too"
        );
    }

    /// The clean, never-aliased case that actually matters for Stage 2: a
    /// value built, passed through a genuinely fresh-constructing call
    /// (`bump`), and never touched again -- the exact `main`-level shape
    /// Stage 2 needs to recognize as safe for headerless allocation. A
    /// single-shot version of this fuses `bump` straight into `main`
    /// (found directly, the same way as the test right above) -- looping
    /// it forces `bump` to survive as a real, separately-analyzed call.
    #[test]
    fn a_locally_constructed_value_passed_only_to_a_pure_borrow_is_never_aliased() {
        let src = "
            struct Boxed { v: i32, tag: [i32; 1] }
            fn bump(a: Boxed) -> Boxed { Boxed(v: a.v + 1, tag: [a.tag[0]]) }
            extern fn opaque_sink(b: Boxed) -> i32;
            fn main() -> i32 {
                let mut acc = 0;
                for i in 0..3 {
                    let a = Boxed(v: i, tag: [0]);
                    acc = acc + opaque_sink(bump(a));
                };
                acc
            }
            ";
        let program = optimized_cps(src);
        let summary = analyze(&program);
        let var = first_struct_var(&program, "main");
        let f = program.funcs.iter().find(|f| f.def.name == "main").unwrap();
        assert!(
            !value_is_ever_aliased(var, &f.def.body, &summary),
            "`a` is only ever borrowed by `bump` (proven never-aliased at \
             its own parameter 0) and never used again -- must not be \
             marked aliased"
        );
    }
}

/// `affine_struct_vars` — the actual Stage 2 decision: both conditions
/// (never aliased, no cascade-worthy field) required together.
mod affine_eligibility {
    use super::*;
    use cleave::alias_analysis::{affine_struct_vars, analyze, analyze_identity};
    use cleave::cps::collect_struct_schemas;
    use cleave::refcount::{
        collect_constructed_struct_names, collect_extern_boundary_struct_names,
        collect_field_mutated_struct_names,
    };

    fn affine_vars(src: &str) -> (cleave::cps::CpsProgram, std::collections::HashSet<cleave::cps::CVar>) {
        let program = optimized_cps(src);
        let summary = analyze(&program);
        let identity_summary = analyze_identity(&program);
        let (compiled, _) = cleave::driver::compile(vec![("t.cleave".to_string(), src.to_string())], &[]);
        let ast_program = compiled.unwrap();
        let struct_schemas = collect_struct_schemas(&ast_program);
        let mlir_types = cleave::cps::collect_mlir_types(&ast_program);
        let constructed = collect_constructed_struct_names(&program);
        let field_mutated = collect_field_mutated_struct_names(&program);
        let extern_boundary = collect_extern_boundary_struct_names(&program);
        let region_local = cleave::region_analysis::find_region_local_functions(&program);
        let affine = affine_struct_vars(
            &program,
            &summary,
            &identity_summary,
            &struct_schemas,
            &mlir_types,
            &constructed,
            &field_mutated,
            &extern_boundary,
            &region_local,
        );
        (program, affine)
    }

    fn nth_struct_var(program: &cleave::cps::CpsProgram, f_name: &str, n: usize) -> cleave::cps::CVar {
        fn walk(expr: &cleave::cps::CExpr, out: &mut Vec<cleave::cps::CVar>) {
            use cleave::cps::{CExpr, PrimOp};
            match expr {
                CExpr::LetPrim { var, op, cont, .. } => {
                    if matches!(op, PrimOp::Struct(..)) {
                        out.push(*var);
                    }
                    walk(cont, out);
                }
                CExpr::App { .. } => {}
                CExpr::If { then_branch, else_branch, .. } => {
                    walk(then_branch, out);
                    walk(else_branch, out);
                }
                CExpr::Fix { defs, body } => {
                    for d in defs {
                        walk(&d.body, out);
                    }
                    walk(body, out);
                }
            }
        }
        let f = program.funcs.iter().find(|f| f.def.name == f_name).unwrap();
        let mut out = Vec::new();
        walk(&f.def.body, &mut out);
        out[n]
    }

    /// The real `b = bump(b)` shape (`doc/backlog.md`) -- looped so `bump`
    /// survives as a real call: `a`, constructed fresh each iteration,
    /// never aliased, no field of its own worth cascading into, must be
    /// affine-eligible.
    #[test]
    fn the_bump_shaped_construction_is_affine_eligible() {
        let src = "
            struct Boxed { v: i32, tag: [i32; 1] }
            fn bump(a: Boxed) -> Boxed { Boxed(v: a.v + 1, tag: [a.tag[0]]) }
            extern fn opaque_sink(b: Boxed) -> i32;
            fn main() -> i32 {
                let mut acc = 0;
                for i in 0..3 {
                    let a = Boxed(v: i, tag: [0]);
                    acc = acc + opaque_sink(bump(a));
                };
                acc
            }
            ";
        let (program, affine) = affine_vars(src);
        let a = nth_struct_var(&program, "main", 0);
        assert!(
            affine.contains(&a),
            "`a` is never aliased and `Boxed` has no cascade-worthy field \
             -- must be affine-eligible"
        );
    }

    /// `doc/plan-affine-ownership.md` §13/§14 — a struct embedding another
    /// refcounted field is now affine-eligible too, *as long as it (and
    /// that field's own struct type) is never field-mutated anywhere* — the
    /// occupant of a never-mutated field is fixed forever from its own
    /// single construction site, decidable statically with no flow-
    /// sensitive points-to needed (confirmed on the real target,
    /// `examples/mnist-interop`: zero `PrimOp::FieldStore` in the whole
    /// compiled kernel). `o.inner.v` alone (construct then immediately
    /// project a field) is *always* folded away by this e-graph regardless
    /// of loop bound — a real function call survives instead (mirrors the
    /// `bump` test above, same reason): `read_outer` only ever borrows its
    /// own parameter, so calling it doesn't alias `o` either.
    fn nth_named_struct_var(program: &cleave::cps::CpsProgram, f_name: &str, ty_name: &str, n: usize) -> cleave::cps::CVar {
        fn walk(expr: &cleave::cps::CExpr, ty_name: &str, out: &mut Vec<cleave::cps::CVar>) {
            use cleave::cps::{CExpr, PrimOp};
            match expr {
                CExpr::LetPrim { var, op, cont, .. } => {
                    if let PrimOp::Struct(name, _) = op {
                        if name == ty_name {
                            out.push(*var);
                        }
                    }
                    walk(cont, ty_name, out);
                }
                CExpr::App { .. } => {}
                CExpr::If { then_branch, else_branch, .. } => {
                    walk(then_branch, ty_name, out);
                    walk(else_branch, ty_name, out);
                }
                CExpr::Fix { defs, body } => {
                    for d in defs {
                        walk(&d.body, ty_name, out);
                    }
                    walk(body, ty_name, out);
                }
            }
        }
        let f = program.funcs.iter().find(|f| f.def.name == f_name).unwrap();
        let mut out = Vec::new();
        walk(&f.def.body, ty_name, &mut out);
        out[n]
    }

    #[test]
    fn a_never_mutated_struct_embedding_a_refcounted_field_is_affine_eligible() {
        let src = "
            struct Inner { v: i32, tag: [i32; 1] }
            struct Outer { inner: Inner }
            fn read_outer(o: Outer) -> i32 { o.inner.v }
            extern fn opaque_sink(v: i32) -> i32;
            fn main() -> i32 {
                let mut acc = 0;
                for i in 0..3 {
                    let o = Outer(inner: Inner(v: i, tag: [0]));
                    acc = acc + opaque_sink(read_outer(o));
                };
                acc
            }
            ";
        let (program, affine) = affine_vars(src);
        let outer = nth_named_struct_var(&program, "main", "Outer", 0);
        assert!(
            affine.contains(&outer),
            "`Outer` embeds `Inner`, but neither is ever field-mutated -- \
             must now be affine-eligible (`doc/plan-affine-ownership.md` \
             §13)"
        );
        let inner = nth_named_struct_var(&program, "main", "Inner", 0);
        assert!(
            affine.contains(&inner),
            "`Inner`'s own construction is never aliased on its own merits \
             either -- must be affine-eligible independently of being \
             embedded"
        );
        let field_affine = cleave::alias_analysis::field_affine_positions(&program, &affine);
        assert_eq!(
            field_affine.get(&("Outer".to_string(), 0)),
            Some(&true),
            "`Outer`'s own field 0 (`inner`) is always constructed from an \
             already-affine value at every one of `Outer`'s own \
             construction sites -- `mlir_lower.rs::lower_release_pool_\
             cascade` needs this to know the cascade into `inner` can also \
             skip the header"
        );
    }

    /// The one case that must still be excluded, precisely because it's
    /// the one this analysis genuinely can't decide statically without the
    /// flow-sensitive points-to `doc/plan-affine-ownership.md` §14.4 scopes
    /// as a separate, harder phase: a field that *is* reassigned somewhere.
    #[test]
    fn a_struct_with_a_field_mutated_refcounted_field_is_never_affine_eligible() {
        let src = "
            struct Inner { v: i32, tag: [i32; 1] }
            struct Outer { inner: Inner }
            fn read_outer(o: Outer) -> i32 { o.inner.v }
            extern fn opaque_sink(v: i32) -> i32;
            fn replace_inner(mut o: Outer, new_inner: Inner) -> Outer {
                o.inner = new_inner;
                o
            }
            fn main() -> i32 {
                let mut acc = 0;
                for i in 0..3 {
                    let o = Outer(inner: Inner(v: i, tag: [0]));
                    let o2 = replace_inner(o, Inner(v: i + 1, tag: [0]));
                    acc = acc + opaque_sink(read_outer(o2));
                };
                acc
            }
            ";
        let (program, affine) = affine_vars(src);
        let outer = nth_named_struct_var(&program, "main", "Outer", 0);
        assert!(
            !affine.contains(&outer),
            "`Outer`'s own `inner` field is reassigned by `replace_inner` -- \
             its own occupant isn't fixed from construction, so this must \
             stay excluded (no flow-sensitive points-to built yet)"
        );
    }

    /// A struct with no cascade-worthy field, but genuinely aliased, must
    /// still be excluded -- the aliasing condition alone is enough to
    /// reject it, independent of field shape.
    #[test]
    fn an_aliased_struct_with_no_cascade_fields_is_not_affine_eligible() {
        let src = "
            struct Boxed { v: i32, tag: [i32; 1] }
            struct Wrapper { inner: Boxed }
            extern fn opaque_sink(w: Wrapper) -> i32;
            extern fn opaque_sink2(b: Boxed) -> i32;
            fn main() -> i32 {
                let a = Boxed(v: 1, tag: [0]);
                let w = Wrapper(inner: a);
                opaque_sink(w) + opaque_sink2(a)
            }
            ";
        let (program, affine) = affine_vars(src);
        let a = nth_struct_var(&program, "main", 0);
        assert!(
            !affine.contains(&a),
            "`a` is embedded into `w` and read again afterward -- must not \
             be affine-eligible"
        );
    }

    /// The resumption parameter of the first call to `callee_name` found in
    /// `f_name`'s own body -- the `CVar` a real call's own result is bound
    /// to (`let a2 = wrap(a);`'s own `a2`), distinct from any `PrimOp::
    /// Struct` site `nth_struct_var` finds.
    fn resumption_var_of_call(
        program: &cleave::cps::CpsProgram,
        f_name: &str,
        callee_name: &str,
    ) -> cleave::cps::CVar {
        fn walk(expr: &cleave::cps::CExpr, callee_name: &str) -> Option<cleave::cps::CVar> {
            use cleave::cps::{CExpr, CVal};
            match expr {
                CExpr::LetPrim { cont, .. } => walk(cont, callee_name),
                CExpr::App { .. } => None,
                CExpr::If { then_branch, else_branch, .. } => {
                    walk(then_branch, callee_name).or_else(|| walk(else_branch, callee_name))
                }
                CExpr::Fix { defs, body } => {
                    if let ([def], CExpr::App { func: CVal::Label(callee), .. }) =
                        (defs.as_slice(), body.as_ref())
                    {
                        if callee == callee_name {
                            if let [p] = def.params.as_slice() {
                                return Some(*p);
                            }
                        }
                    }
                    defs.iter()
                        .find_map(|d| walk(&d.body, callee_name))
                        .or_else(|| walk(body, callee_name))
                }
            }
        }
        let f = program.funcs.iter().find(|f| f.def.name == f_name).unwrap();
        walk(&f.def.body, callee_name)
            .unwrap_or_else(|| panic!("no call to {callee_name} found in {f_name}"))
    }

    /// The identity-based propagation rule, added alongside `analyze_
    /// identity`'s own join-point generalization (`doc/backlog.md`'s
    /// "examples/complex.cleave --run" entry has the full story):
    /// `Display::display<Complex<T>>`'s real shape is a function that
    /// constructs *nothing* of its own, just hands one of its own
    /// parameters straight back (through an if/else join, proven by
    /// `analyze_identity`) -- if the caller's own argument at that position
    /// is *already* affine, the call's own resumption denotes that *same*
    /// allocation, not a fresh one, and must be affine too, or its release
    /// disagrees with whatever allocator actually backed it (`cleave_
    /// release` on a `cleave_alloc_pool`-only pointer -- a real, confirmed
    /// type-confused free, not hypothetical).
    #[test]
    fn a_resumption_forwarding_an_already_affine_argument_through_an_identity_shaped_callee_is_affine_too() {
        let src = "
            struct Acc { v: i32 }
            fn wrap(a: Acc) -> Acc { a }
            extern fn opaque_sink(x: i32) -> i32;
            fn main() -> i32 {
                let a = Acc(v: 1);
                let a2 = wrap(a);
                opaque_sink(a2.v)
            }
            ";
        let (program, affine) = affine_vars(src);
        let a = nth_struct_var(&program, "main", 0);
        let a2 = resumption_var_of_call(&program, "main", "wrap");
        assert!(affine.contains(&a), "`a` is never aliased -- must be affine-eligible on its own");
        assert!(
            affine.contains(&a2),
            "`wrap` is identity-shaped (hands `a` straight back) and its \
             caller's own argument `a` is already affine -- `a2`, the \
             resumption receiving `wrap`'s result, denotes that exact same \
             allocation and must be affine too"
        );
    }

    /// The `i`-th parameter of the first loop/`if`-join `Fix`-def found in
    /// `f_name`'s own body (`def.carried_types.is_some()`) -- the loop-
    /// carried `CVar` `refcount::insert_refcounting` actually targets with
    /// its own `Release`, distinct from any `PrimOp::Struct` site
    /// `nth_struct_var` finds.
    fn nth_loop_carried_var(program: &cleave::cps::CpsProgram, f_name: &str, i: usize) -> cleave::cps::CVar {
        fn walk(expr: &cleave::cps::CExpr, i: usize) -> Option<cleave::cps::CVar> {
            match expr {
                cleave::cps::CExpr::LetPrim { cont, .. } => walk(cont, i),
                cleave::cps::CExpr::App { .. } => None,
                cleave::cps::CExpr::If { then_branch, else_branch, .. } => {
                    walk(then_branch, i).or_else(|| walk(else_branch, i))
                }
                cleave::cps::CExpr::Fix { defs, body } => defs
                    .iter()
                    .find(|d| d.carried_types.is_some())
                    .map(|d| d.params[i])
                    .or_else(|| defs.iter().find_map(|d| walk(&d.body, i)))
                    .or_else(|| walk(body, i)),
            }
        }
        let f = program.funcs.iter().find(|f| f.def.name == f_name).unwrap();
        walk(&f.def.body, i).unwrap()
    }

    /// The real, genuinely loop-*carried* shape (`b = bump(b)`, not a fresh
    /// per-iteration construction like the test above) -- `doc/plan-affine-
    /// ownership.md` §11.4's own confirmed `STATUS_HEAP_CORRUPTION`. Both
    /// the loop's own *entry* argument (the construction before the loop)
    /// and the loop-carried parameter *itself* (never a `PrimOp::Struct`
    /// site, the actual `CVar` `refcount::insert_refcounting` releases)
    /// must be affine-eligible -- unlockable at all only once `analyze()`
    /// stopped treating a call to a local loop label as automatically
    /// aliased (this same module's own doc comment on `analyze`).
    #[test]
    fn a_genuinely_loop_carried_construction_and_its_own_carried_parameter_are_both_affine_eligible() {
        let src = "
            struct Boxed { v: i32, tag: [i32; 1] }
            fn bump(a: Boxed) -> Boxed { Boxed(v: a.v + 1, tag: [a.tag[0]]) }
            fn main() -> i32 {
                let mut b = Boxed(v: 0, tag: [0]);
                for i in 0..2000 {
                    b = bump(b);
                };
                b.v
            }
            ";
        let (program, affine) = affine_vars(src);
        let entry = nth_struct_var(&program, "main", 0);
        assert!(
            affine.contains(&entry),
            "the construction feeding the loop's own entry argument is \
             never aliased -- must be affine-eligible now that `analyze()` \
             can see what the loop itself does with its own parameter"
        );
        let carried = nth_loop_carried_var(&program, "main", 1);
        assert!(
            affine.contains(&carried),
            "the loop's own carried parameter -- the exact `CVar` \
             `insert_refcounting` releases every iteration -- must be \
             affine-eligible too, or its release still reads a header \
             that was never written on the iterations backed by the pool"
        );
    }

    /// The genuine mutual-dependency gap `collect_affine_carried_params`'s
    /// own doc comment has the full story on: unlike the test above (`b =
    /// bump(b)`, a fresh `PrimOp::Struct` construction each iteration), a
    /// carried value threaded each iteration through an *identity-shaped*
    /// real call (`Display::display<Complex<T>>`'s own real shape,
    /// `doc/backlog.md`'s "examples/complex.cleave" entry) creates a real
    /// deadlock the original per-call-site rule could never resolve: the
    /// back-edge argument is that call's own resumption parameter, only
    /// provably affine once the carried parameter itself already is -- and
    /// vice versa. Confirmed to fail before this fix (`entry affine =
    /// true`, `carried affine = false`, via a dedicated debug probe) even
    /// though the underlying identity fact was already correct.
    #[test]
    fn a_carried_parameter_threaded_through_an_identity_shaped_real_call_each_iteration_is_affine_too() {
        let src = "
            struct Boxed { v: i32, tag: [i32; 1] }
            extern fn opaque_sink(x: i32) -> i32;
            fn touch1(a: Boxed) -> i32 { a.v }
            fn touch2(a: Boxed) -> i32 { a.v + 1 }
            fn thread_through(cond: bool, a: Boxed) -> Boxed {
                if cond {
                    opaque_sink(touch1(a));
                    a
                } else {
                    opaque_sink(touch2(a));
                    a
                }
            }
            fn main() -> i32 {
                let mut b = Boxed(v: 0, tag: [0]);
                for i in 0..2000 {
                    b = thread_through(i < 1000, b);
                };
                b.v
            }
            ";
        let (program, affine) = affine_vars(src);
        let entry = nth_struct_var(&program, "main", 0);
        assert!(affine.contains(&entry), "the entry construction is never aliased -- must be affine-eligible");
        let carried = nth_loop_carried_var(&program, "main", 1);
        assert!(
            affine.contains(&carried),
            "the loop's own carried parameter is threaded through `thread_through` \
             (identity-shaped at that position) every iteration -- it denotes the \
             exact same allocation as the entry argument throughout and must be \
             affine-eligible too, despite the mutual dependency between it and \
             `thread_through`'s own resumption parameter"
        );
    }
}

/// The whole reason `occurs_in` deliberately excludes `Retain`/`Release`
/// from its own occurs-check (`alias_analysis.rs`'s own doc comment on
/// that function): `affine_struct_vars` must give the *same* answer
/// whether it runs on the pre-refcounting CPS it was designed against, or
/// on the CPS `refcount::insert_refcounting` has already instrumented --
/// which is what `mlir_lower.rs::lower_program` actually receives, since
/// threading a *new* parameter through its own four call sites (touching
/// 18 files, `main.rs` and 14 test files included) was judged too
/// invasive for this landing. Without the fix, `a`'s own `Release`
/// (inserted by `insert_refcounting` at its true last use) would be
/// wrongly counted as "`a` occurs again", making it look aliased.
#[test]
fn affine_eligibility_gives_the_same_answer_before_and_after_insert_refcounting() {
    use cleave::alias_analysis::{affine_struct_vars, analyze, analyze_identity};
    use cleave::cps::collect_struct_schemas;
    use cleave::refcount::{
        collect_constructed_struct_names, collect_extern_boundary_struct_names,
        collect_field_mutated_struct_names, insert_refcounting,
    };

    let src = "
        struct Boxed { v: i32, tag: [i32; 1] }
        fn bump(a: Boxed) -> Boxed { Boxed(v: a.v + 1, tag: [a.tag[0]]) }
        extern fn opaque_sink(b: Boxed) -> i32;
        fn main() -> i32 {
            let mut acc = 0;
            for i in 0..3 {
                let a = Boxed(v: i, tag: [0]);
                acc = acc + opaque_sink(bump(a));
            };
            acc
        }
        ";
    let (compiled, _) = cleave::driver::compile(vec![("t.cleave".to_string(), src.to_string())], &[]);
    let ast_program = compiled.unwrap();
    let struct_schemas = collect_struct_schemas(&ast_program);
    let mlir_types = cleave::cps::collect_mlir_types(&ast_program);

    let pre_refcounting = optimized_cps(src);
    let summary = analyze(&pre_refcounting);
    let constructed = collect_constructed_struct_names(&pre_refcounting);
    let field_mutated = collect_field_mutated_struct_names(&pre_refcounting);
    let extern_boundary = collect_extern_boundary_struct_names(&pre_refcounting);
    let escaping = cleave::escape::escaping_struct_vars(&pre_refcounting);
    let region_local = cleave::region_analysis::find_region_local_functions(&pre_refcounting);

    // The same `a`, findable by identical CVar numbering before and after
    // `insert_refcounting` -- that pass only ever *adds* new `Retain`/
    // `Release` bindings of its own, it never renumbers an existing one.
    fn first_struct_var(program: &cleave::cps::CpsProgram, f_name: &str) -> cleave::cps::CVar {
        fn walk(expr: &cleave::cps::CExpr) -> Option<cleave::cps::CVar> {
            use cleave::cps::{CExpr, PrimOp};
            match expr {
                CExpr::LetPrim { var, op, cont, .. } => {
                    if matches!(op, PrimOp::Struct(..)) { Some(*var) } else { walk(cont) }
                }
                CExpr::App { .. } => None,
                CExpr::If { then_branch, else_branch, .. } => {
                    walk(then_branch).or_else(|| walk(else_branch))
                }
                CExpr::Fix { defs, body } => {
                    defs.iter().find_map(|d| walk(&d.body)).or_else(|| walk(body))
                }
            }
        }
        let f = program.funcs.iter().find(|f| f.def.name == f_name).unwrap();
        walk(&f.def.body).unwrap()
    }
    let a = first_struct_var(&pre_refcounting, "main");

    let identity_summary_before = analyze_identity(&pre_refcounting);
    let affine_before = affine_struct_vars(
        &pre_refcounting,
        &summary,
        &identity_summary_before,
        &struct_schemas,
        &mlir_types,
        &constructed,
        &field_mutated,
        &extern_boundary,
        &region_local,
    );
    assert!(affine_before.contains(&a), "must be affine BEFORE insert_refcounting");

    let post_refcounting = insert_refcounting(pre_refcounting, &struct_schemas, &mlir_types, &escaping);
    let identity_summary_after = analyze_identity(&post_refcounting);
    let affine_after = affine_struct_vars(
        &post_refcounting,
        &summary,
        &identity_summary_after,
        &struct_schemas,
        &mlir_types,
        &constructed,
        &field_mutated,
        &extern_boundary,
        &region_local,
    );
    assert!(
        affine_after.contains(&a),
        "must give the SAME answer (affine) AFTER insert_refcounting has \
         instrumented the program -- this is the exact scenario `mlir_\
         lower.rs::lower_program` runs in"
    );
}
