//! Real, end-to-end proof for `cleave::refcount::insert_refcounting`
//! (`doc/hld.md`'s own "Memory management" section, Phase 0's struct/
//! descriptor half) — each test JIT-executes a real compiled program and
//! checks its actual numeric result, not just that the generated CPS/MLIR
//! *looks* plausible. Several of these are direct regression tests for
//! real bugs found via `examples/digits-interop` (a genuine
//! `STATUS_HEAP_CORRUPTION`/`STATUS_ACCESS_VIOLATION`, not a hypothetical)
//! — see `refcount.rs`'s own module doc comment for the full story behind
//! each one; a bare "doesn't crash" is itself the meaningful assertion for
//! those (a premature or missing release manifests as corruption, not a
//! wrong-but-plausible number, especially at the tiny scale these tests
//! run at — real correctness needs `examples/digits-interop`'s own actual
//! training run, already verified separately, not repeatable here).

use cleave::cps::{
    CExpr, CpsProgram, PrimOp, collect_mlir_types, collect_struct_schemas, collect_units,
    convert_program,
};
use cleave::driver::compile;
use cleave::egraph::optimize_program;
use cleave::infer::Ty;
use cleave::mlir_lower::lower_program;
use cleave::pipeline::check_type_errors;
use cleave::refcount::insert_refcounting;
use cleave::registry::Registry;
use melior::Context;
use melior::dialect::DialectRegistry;
use melior::ir::operation::OperationLike;
use melior::pass;
use melior::utility::register_all_dialects;

fn context() -> Context {
    let dialect_registry = DialectRegistry::new();
    register_all_dialects(&dialect_registry);
    let context = Context::new();
    context.append_dialect_registry(&dialect_registry);
    context.load_all_available_dialects();
    context
}

/// The shared front half of `run_i32` below — real pipeline, stopping right
/// after `insert_refcounting` instead of going on to JIT — for a test that
/// needs to inspect the *inserted* `Retain`/`Release` calls directly rather
/// than only observe whether the compiled program crashes.
fn refcounted_cps(src: &str) -> CpsProgram {
    let (result, _sources) = compile(vec![("test.cleave".to_string(), src.to_string())], &[]);
    let program = result.unwrap_or_else(|e| panic!("compile failed: {e:?}"));
    let registry = Registry::build(&program);
    if let Err(diags) = check_type_errors(&program, &registry) {
        panic!("type check failed: {diags:?}");
    }
    let units = collect_units(&program, &registry);
    let cps_program = convert_program(units);
    let cps_program = cleave::cps::eliminate_dead_code(cps_program);
    let (cps_program, _) = optimize_program(cps_program, &registry, false);
    let cps_program = cleave::cps::eliminate_dead_code(cps_program);
    let struct_schemas = collect_struct_schemas(&program);
    let mlir_types = collect_mlir_types(&program);
    insert_refcounting(cps_program, &struct_schemas, &mlir_types)
}

/// Every struct name any `Retain`/`Release` in `program` targets — walks
/// every top-level function's own body, recursively through every nested
/// `Fix`/`If`, mirroring `region_analysis.rs`'s/`refcount.rs`'s own
/// established "plain recursive `CExpr` walk" shape for this same kind of
/// whole-program structural fact.
fn retained_or_released_struct_names(program: &CpsProgram) -> std::collections::HashSet<String> {
    let mut names = std::collections::HashSet::new();
    for f in &program.funcs {
        collect_rc_targets(&f.def.body, &mut names);
    }
    names
}

fn collect_rc_targets(expr: &CExpr, names: &mut std::collections::HashSet<String>) {
    match expr {
        CExpr::LetPrim { op, cont, .. } => {
            if let PrimOp::Retain(ty) | PrimOp::Release(ty) = op {
                if let Ty::Con(name) | Ty::App(name, _) = ty {
                    names.insert(name.clone());
                }
            }
            collect_rc_targets(cont, names);
        }
        CExpr::App { .. } => {}
        CExpr::If {
            then_branch,
            else_branch,
            ..
        } => {
            collect_rc_targets(then_branch, names);
            collect_rc_targets(else_branch, names);
        }
        CExpr::Fix { defs, body } => {
            for d in defs {
                collect_rc_targets(&d.body, names);
            }
            collect_rc_targets(body, names);
        }
    }
}

/// The real, refcounted MLIR module's own printed text for `src` — one step
/// further than `refcounted_cps` above: also runs `mlir_lower::lower_
/// program` (real pipeline order, `pipeline.rs::build_optimized_cps`'s own
/// sequencing) and hands back the verified module's own text, for a test
/// that needs to inspect what `lower_release_cascade` (`mlir_lower.rs`)
/// actually generated for a CPS-level `Release` — invisible to `refcounted_
/// cps`'s own CPS-level structural checks above (`retained_or_released_
/// struct_names`, `count_retains_for`), since the cascade *into* a struct's
/// own fields is built entirely during MLIR lowering, from a single CPS-
/// level `Release` node, never itself represented as further `Retain`/
/// `Release` primops in the CPS term.
fn refcounted_mlir_text(context: &Context, src: &str) -> String {
    let (result, _sources) = compile(vec![("test.cleave".to_string(), src.to_string())], &[]);
    let program = result.unwrap_or_else(|e| panic!("compile failed: {e:?}"));
    let registry = Registry::build(&program);
    if let Err(diags) = check_type_errors(&program, &registry) {
        panic!("type check failed: {diags:?}");
    }
    let cps_program = refcounted_cps(src);
    let mlir_types = collect_mlir_types(&program);
    let struct_schemas = collect_struct_schemas(&program);
    let module = lower_program(context, &cps_program, &mlir_types, struct_schemas);
    assert!(
        module.as_operation().verify(),
        "generated MLIR module failed verification"
    );
    module.as_operation().to_string()
}

/// How many `Retain(ty)` calls anywhere in `program` name struct
/// `struct_name` — used to check `insert_refcounting` retained *every*
/// element of a construction, not just recognized the struct type exists
/// somewhere.
fn count_retains_for(program: &CpsProgram, struct_name: &str) -> usize {
    let mut count = 0;
    for f in &program.funcs {
        count_retains_in(&f.def.body, struct_name, &mut count);
    }
    count
}

fn count_retains_in(expr: &CExpr, struct_name: &str, count: &mut usize) {
    match expr {
        CExpr::LetPrim { op, cont, .. } => {
            if let PrimOp::Retain(ty) = op {
                if matches!(ty, Ty::Con(n) | Ty::App(n, _) if n == struct_name) {
                    *count += 1;
                }
            }
            count_retains_in(cont, struct_name, count);
        }
        CExpr::App { .. } => {}
        CExpr::If {
            then_branch,
            else_branch,
            ..
        } => {
            count_retains_in(then_branch, struct_name, count);
            count_retains_in(else_branch, struct_name, count);
        }
        CExpr::Fix { defs, body } => {
            for d in defs {
                count_retains_in(&d.body, struct_name, count);
            }
            count_retains_in(body, struct_name, count);
        }
    }
}

/// Compiles `src` through the *real* pipeline (`pipeline.rs::
/// build_optimized_cps`'s own exact sequence: CPS conversion, dead-code
/// elimination, e-graph optimization, a second dead-code sweep, then
/// `insert_refcounting`), lowers to the `llvm` dialect, and JIT-invokes
/// `main`, returning its own `i32` result.
fn run_i32(src: &str) -> i32 {
    run_i32_with_extra_symbols(src, &[])
}

/// Like `run_i32`, but registers extra runtime symbols alongside the base
/// set — needed for a program that reaches `stdlib/dynarray/dynarray.
/// cleave` (`use dynarray;` eagerly compiles every one of its own six
/// non-generic `RawBuffer<T>` impls regardless of which widths the test
/// program itself actually uses, `cps.rs`'s own "non-generic impl" branch —
/// mirrors `tests/mlir_lower.rs::run_i32_with_dynarray_symbols`'s identical
/// reasoning).
fn run_i32_with_extra_symbols(src: &str, extra_symbols: &[(&str, *mut ())]) -> i32 {
    let context = context();
    let (result, _sources) = compile(vec![("test.cleave".to_string(), src.to_string())], &[]);
    let program = result.unwrap_or_else(|e| panic!("compile failed: {e:?}"));
    let registry = Registry::build(&program);
    if let Err(diags) = check_type_errors(&program, &registry) {
        panic!("type check failed: {diags:?}");
    }
    let units = collect_units(&program, &registry);
    let cps_program = convert_program(units);
    let cps_program = cleave::cps::eliminate_dead_code(cps_program);
    let (cps_program, _) = optimize_program(cps_program, &registry, false);
    let cps_program = cleave::cps::eliminate_dead_code(cps_program);
    let struct_schemas = collect_struct_schemas(&program);
    let mlir_types = collect_mlir_types(&program);
    let cps_program = insert_refcounting(cps_program, &struct_schemas, &mlir_types);

    let mlir_types = collect_mlir_types(&program);
    let struct_schemas = collect_struct_schemas(&program);
    let mut module = lower_program(&context, &cps_program, &mlir_types, struct_schemas);
    assert!(
        module.as_operation().verify(),
        "generated MLIR module failed verification"
    );

    let pass_manager = pass::PassManager::new(&context);
    pass_manager.add_pass(pass::conversion::create_scf_to_control_flow());
    pass_manager.add_pass(pass::conversion::create_to_llvm());
    pass_manager.add_pass(pass::conversion::create_reconcile_unrealized_casts());
    pass_manager
        .run(&mut module)
        .expect("lowering to the llvm dialect must succeed");

    let engine = melior::ExecutionEngine::new(&module, 2, &[], false, false);
    // SAFETY: each `cleave_rt::*` pointer is a real, valid `extern "C" fn`,
    // live for the process's whole lifetime — mirrors `pipeline.rs::
    // register_cleave_rt_symbols`'s own registration.
    unsafe {
        engine.register_symbol("cleave_alloc", cleave_rt::cleave_alloc as *mut ());
        engine.register_symbol("cleave_alloc_rc", cleave_rt::cleave_alloc_rc as *mut ());
        engine.register_symbol("cleave_retain", cleave_rt::cleave_retain as *mut ());
        engine.register_symbol("cleave_release", cleave_rt::cleave_release as *mut ());
        engine.register_symbol("cleave_release_void", cleave_rt::cleave_release_void as *mut ());
        engine.register_symbol("cleave_alloc_local", cleave_rt::cleave_alloc_local as *mut ());
        engine.register_symbol("cleave_region_enter", cleave_rt::cleave_region_enter as *mut ());
        engine.register_symbol("cleave_region_exit", cleave_rt::cleave_region_exit as *mut ());
        for (name, ptr) in extra_symbols {
            engine.register_symbol(name, *ptr);
        }
    }
    let mut out: i32 = -1;
    // SAFETY: `out` is a live, correctly-aligned `i32` on the stack for the
    // duration of this call.
    unsafe {
        engine
            .invoke_packed("main", &mut [&mut out as *mut i32 as *mut ()])
            .expect("JIT invocation must succeed — a real crash here (STATUS_HEAP_CORRUPTION/STATUS_ACCESS_VIOLATION) is exactly the failure mode a wrong retain/release insertion produces");
    }
    out
}

/// The simplest case: a struct constructed and never used again must be
/// released without disturbing anything else in scope.
#[test]
fn an_unused_locally_constructed_struct_is_released_cleanly() {
    let src = "struct Point { x: i32, y: i32 }
    fn make_unused() -> i32 {
        let p = Point(x: 1, y: 2);
        42
    }
    fn main() -> i32 { make_unused() }";
    assert_eq!(run_i32(src), 42);
}

/// A struct returned from a function must *not* be released before
/// returning — the caller reads its fields afterward.
#[test]
fn a_returned_struct_is_not_released_and_its_fields_read_correctly_afterward() {
    let src = "struct Point { x: i32, y: i32 }
    fn make_point() -> Point {
        let p = Point(x: 1, y: 2);
        p
    }
    fn main() -> i32 {
        let p = make_point();
        p.x + p.y
    }";
    assert_eq!(run_i32(src), 3);
}

/// A struct passed as a *borrowed* parameter to two separate calls must
/// remain valid for both — parameters are never released by the callee
/// (`insert_refcounting_fn`'s own doc comment).
#[test]
fn a_struct_borrowed_by_two_separate_calls_stays_valid_for_both() {
    let src = "struct Point { x: i32, y: i32 }
    fn sum(p: Point) -> i32 { p.x + p.y }
    fn main() -> i32 {
        let p = Point(x: 3, y: 4);
        let a = sum(p);
        let b = sum(p);
        a + b
    }";
    assert_eq!(run_i32(src), 14);
}

/// A loop constructing a fresh struct every iteration, reassigned via
/// `let mut` — each iteration's own struct must be released once replaced,
/// without corrupting the next one.
#[test]
fn a_freshly_constructed_struct_reassigned_every_loop_iteration_is_released_correctly() {
    let src = "struct Point { x: i32, y: i32 }
    fn main() -> i32 {
        let mut total = 0;
        for i in 0..100 {
            let p = Point(x: i, y: i);
            total = total + p.x + p.y;
        };
        total
    }";
    assert_eq!(run_i32(src), 9900);
}

/// Storing an *existing* struct-typed value into another struct's own
/// field (construction-time embedding, `Line(a: p1, b: p2)`) creates a
/// second, independent reference — both the original bindings (`p1`/`p2`)
/// and the new container (`l`) must remain independently valid and
/// readable. A missing retain here is exactly the bug `rewrite_body`'s own
/// "retain-on-construction" logic exists to prevent (a dangling read that
/// may not manifest as a wrong value immediately, since freed memory isn't
/// always overwritten right away — the real assertion here is `main`
/// returning without corruption at all).
#[test]
fn embedding_an_existing_struct_into_a_new_ones_own_field_retains_it_correctly() {
    let src = "struct Point { x: i32, y: i32 }
    struct Line { a: Point, b: Point }
    fn make_line() -> Line {
        let p1 = Point(x: 1, y: 2);
        let p2 = Point(x: 3, y: 4);
        Line(a: p1, b: p2)
    }
    fn main() -> i32 {
        let l = make_line();
        l.a.x + l.a.y + l.b.x + l.b.y
    }";
    assert_eq!(run_i32(src), 10);
}

/// A struct value carried across a loop (`let mut`, reassigned inside)
/// *through a real function call* every iteration — not just a local
/// reconstruction (already covered above). Regression test for the first
/// real `STATUS_HEAP_CORRUPTION` found via `examples/digits-interop`: the
/// value returned by the call and the loop's own previous carried value
/// must not collide, and the carried value must survive across many
/// iterations without being released early.
#[test]
fn a_struct_carried_through_a_loop_via_a_real_function_call_stays_correct_across_iterations() {
    let src = "struct Point { x: i32, y: i32 }
    fn bump(p: Point) -> Point {
        Point(x: p.x + 1, y: p.y + 1)
    }
    fn main() -> i32 {
        let mut p = Point(x: 0, y: 0);
        for i in 0..20 {
            p = bump(p);
        };
        p.x + p.y
    }";
    assert_eq!(run_i32(src), 40);
}

/// Regression test for the second and third real `STATUS_HEAP_CORRUPTION`
/// bugs found via `examples/digits-interop` (`train_and_evaluate`'s own
/// `opt`): a fresh, owned struct constructed *outside* two nested loops,
/// referenced only *inside* the innermost one (through an intervening real
/// call's own resumption continuation, exactly like `Optimizer::init_state`
/// sitting between `opt`'s own construction and the training loop) — must
/// survive every iteration of *both* loops and be released exactly once,
/// after the outer loop is genuinely done. A premature release (the
/// second bug: the inner loop releasing it at the end of its own first
/// invocation) or a "trampoline" resumption hiding it from `live_set` (the
/// third bug) each reproduced a real crash at this exact shape; a
/// correctness assertion on the final value, not just "doesn't crash",
/// confirms the shared struct's own fields weren't corrupted along the way
/// either.
#[test]
fn a_struct_referenced_only_through_an_intervening_call_survives_two_nested_loops() {
    let src = "struct Config { step: i32 }
    fn advance(c: Config, p: i32) -> i32 { p + c.step }
    fn make_config() -> Config { Config(step: 1) }
    fn main() -> i32 {
        let cfg = make_config();
        let mut total = 0;
        for outer in 0..3 {
            for inner in 0..4 {
                total = advance(cfg, total);
            };
        };
        total
    }";
    assert_eq!(run_i32(src), 12);
}

/// A real, found-by-testing bug (`doc/backlog-done.md`'s own "`is_
/// refcounted` didn't exclude an opaque-handle struct with no real
/// construction site" entry, root-caused against `examples/convex_hull.
/// cleave --run`'s own intermittent `cleave_release: invalid layout:
/// LayoutError` crash): `RawBuf {}` (`stdlib/dynarray/dynarray.cleave`) is
/// an ordinary, untagged, zero-field struct declaration, but its own real
/// values are *never* built via `RawBuf(...)` anywhere — only ever produced
/// by an `extern fn` (`dynarray_alloc_ptr`/`dynarray_alloc_i32`/...), a
/// plain `realloc`-backed pointer with no `RcHeader` in front of it at all.
/// Before this fix, `is_refcounted` was purely type-based, blind to origin,
/// so a `RawBuf`-typed struct field (`DynArray.buf`) still got ordinary
/// `Retain`/`Release` calls inserted around it — reading/writing an
/// `RcHeader` that was never really there, real, non-deterministic memory
/// corruption. Structural, not JIT-execution-based: the corruption itself
/// only manifests probabilistically at runtime (garbage bytes vary run to
/// run), so a single JIT invocation isn't a reliable enough signal on its
/// own — this asserts directly on the *inserted* `Retain`/`Release` calls
/// instead, which is deterministic.
#[test]
fn an_opaque_handle_struct_with_no_real_construction_site_is_never_retained_or_released() {
    let src = r#"
        use dynarray;
        fn main() -> i32 {
            let h: DynArray<i32> = dynarray_new(4);
            h.push(1);
            h.push(2);
            h.get(0) + h.get(1)
        }
        "#;
    let program = refcounted_cps(src);
    let rc_targets = retained_or_released_struct_names(&program);
    assert!(
        !rc_targets.contains("RawBuf"),
        "RawBuf is never constructed via `RawBuf(...)` anywhere -- its own \
         values are always a plain, non-`cleave_alloc_rc`'d pointer from an \
         `extern fn`, so retaining/releasing one reads/writes a header that \
         was never really there; got: {rc_targets:?}"
    );
    // `DynArray` itself is a real, genuinely-constructed struct
    // (`dynarray_new`'s own `DynArray(buf:...,len:...,cap:...)`) -- must
    // stay refcounted normally, proving this fix didn't over-broadly
    // exclude anything.
    assert!(
        rc_targets.contains("DynArray"),
        "DynArray is a real, `cleave_alloc_rc`'d struct -- this fix must not \
         exclude it too; got: {rc_targets:?}"
    );
}

/// A real, found-by-code-inspection bug, one layer beneath the CPS-level
/// fix just above: `is_refcounted`'s own "has a real construction site"
/// exclusion (the fix for the *original* `RawBuf` corruption) is only ever
/// consulted by `insert_refcounting` when deciding whether to emit a *top-
/// level* `Retain`/`Release` for a CPS-level variable — `mlir_lower.rs::
/// lower_release_cascade` (which recurses a struct's own `Release` into its
/// *own fields*, entirely at MLIR-lowering time, never itself represented
/// as further CPS-level `Retain`/`Release` nodes) used to decide whether to
/// recurse into a given field with a much cruder check: "is this a declared
/// struct type", with no awareness that a declared-but-never-constructed
/// one (`DynArray<T>`'s own `buf: RawBuf` field, concretely) has no real
/// `RcHeader` to act on at all. `DynArray` embedded in a further struct
/// (`Wrapper`, mirroring `Network` embedding `Dense`, the real shape
/// `doc/backlog.md`'s own "cleave_release is non-cascading..." item
/// describes) is exactly the shape that exercises this: releasing `Wrapper`
/// cascades into its own `arr: DynArray<i32>` field, which — before this
/// fix — cascaded *again* into `arr`'s own `buf: RawBuf` field, generating
/// a real `cleave_release` call against a raw, non-headered pointer
/// (confirmed directly, before landing the fix: the exact same corruption
/// class `is_refcounted`'s own doc comment already documents as genuinely
/// non-deterministic — roughly a third of the time a visible panic, the
/// rest silent — which is why this is a *structural* MLIR-text check, not
/// an execution-based one: a JIT run not crashing proves nothing here).
#[test]
fn releasing_a_struct_that_embeds_a_dynarray_never_cascades_into_its_own_rawbuf_field() {
    let src = r#"
        use dynarray;
        struct Wrapper { arr: DynArray<i32> }
        fn make_and_discard(cap: i32) -> i32 {
            let d: DynArray<i32> = dynarray_new(cap);
            let w: Wrapper = Wrapper(arr: d);
            w.arr.len
        }
        fn main() -> i32 {
            make_and_discard(4)
        }
        "#;
    let text = refcounted_mlir_text(&context(), src);
    let release_count = text.matches("call @cleave_release").count();
    // The real, fixed count for this exact program: `d` released directly
    // (no cascade — `buf` is correctly excluded), `w` released (cascading
    // into its own live `arr` field, one more release), `w.arr`'s own
    // separately-retained re-read released once more — 4 total. Before
    // this fix, the *same* program generated a 5th `cleave_release` call,
    // nested inside `d`'s own release cascade, targeting `buf`'s own raw
    // `RawBuf` pointer directly (confirmed directly, by temporarily
    // reverting just this fix and re-diffing the generated module text).
    assert!(
        release_count <= 4,
        "expected at most 4 `cleave_release` calls (none of them cascading \
         into RawBuf) -- got {release_count}, suggesting the cascade is \
         once again recursing into a never-constructed struct field:\n{text}"
    );
}

/// The same fix, verified end to end via a real JIT run too (not a
/// reliable *reproduction* of the probabilistic corruption on its own —
/// see the structural test above for that — but a real, additional
/// correctness check: the actual computed values must still be right).
#[test]
fn dynarray_of_primitives_still_computes_correct_values_after_the_rawbuf_fix() {
    let src = r#"
        use dynarray;
        fn main() -> i32 {
            let h: DynArray<i32> = dynarray_new(4);
            h.push(10);
            h.push(20);
            h.push(30);
            h.get(0) + h.get(1) + h.get(2)
        }
        "#;
    // `use dynarray;` eagerly compiles all six non-generic `RawBuffer<T>`
    // impls, not just the `i32` one this program actually uses — every
    // width's own `dynarray_*` symbol needs to be resolvable at JIT link
    // time regardless (`run_i32_with_extra_symbols`'s own doc comment).
    let symbols: &[(&str, *mut ())] = &[
        ("dynarray_alloc_i8", cleave_rt::dynarray_alloc_i8 as *mut ()),
        ("dynarray_grow_i8", cleave_rt::dynarray_grow_i8 as *mut ()),
        ("dynarray_get_i8", cleave_rt::dynarray_get_i8 as *mut ()),
        ("dynarray_set_i8", cleave_rt::dynarray_set_i8 as *mut ()),
        ("dynarray_alloc_i16", cleave_rt::dynarray_alloc_i16 as *mut ()),
        ("dynarray_grow_i16", cleave_rt::dynarray_grow_i16 as *mut ()),
        ("dynarray_get_i16", cleave_rt::dynarray_get_i16 as *mut ()),
        ("dynarray_set_i16", cleave_rt::dynarray_set_i16 as *mut ()),
        ("dynarray_alloc_i32", cleave_rt::dynarray_alloc_i32 as *mut ()),
        ("dynarray_grow_i32", cleave_rt::dynarray_grow_i32 as *mut ()),
        ("dynarray_get_i32", cleave_rt::dynarray_get_i32 as *mut ()),
        ("dynarray_set_i32", cleave_rt::dynarray_set_i32 as *mut ()),
        ("dynarray_alloc_i64", cleave_rt::dynarray_alloc_i64 as *mut ()),
        ("dynarray_grow_i64", cleave_rt::dynarray_grow_i64 as *mut ()),
        ("dynarray_get_i64", cleave_rt::dynarray_get_i64 as *mut ()),
        ("dynarray_set_i64", cleave_rt::dynarray_set_i64 as *mut ()),
        ("dynarray_alloc_f32", cleave_rt::dynarray_alloc_f32 as *mut ()),
        ("dynarray_grow_f32", cleave_rt::dynarray_grow_f32 as *mut ()),
        ("dynarray_get_f32", cleave_rt::dynarray_get_f32 as *mut ()),
        ("dynarray_set_f32", cleave_rt::dynarray_set_f32 as *mut ()),
        ("dynarray_alloc_f64", cleave_rt::dynarray_alloc_f64 as *mut ()),
        ("dynarray_grow_f64", cleave_rt::dynarray_grow_f64 as *mut ()),
        ("dynarray_get_f64", cleave_rt::dynarray_get_f64 as *mut ()),
        ("dynarray_set_f64", cleave_rt::dynarray_set_f64 as *mut ()),
    ];
    assert_eq!(run_i32_with_extra_symbols(src, symbols), 60);
}

/// A real, found-by-testing bug (`doc/backlog-done.md`'s own "an array
/// literal of struct-typed elements got no retain at all" entry,
/// root-caused against `examples/convex_hull.cleave --run`'s own
/// intermittent memory corruption by dumping `--dump-cps-optimized`
/// directly): `[p1, p2, p3]` (`PrimOp::Array`, `ExprKind::ArrayLit`) embeds
/// each element's own pointer directly, the exact same "aliases every
/// argument" shape `PrimOp::Struct` construction already gets a retain
/// for — but `insert_refcounting`'s own `retain_targets` match never
/// included `PrimOp::Array` at all, so every freshly-constructed struct fed
/// into an array literal was released immediately after the array was
/// built, with no retain protecting the array's own now-dangling copy.
/// Structural, not JIT-execution-based, for the identical reason the
/// `RawBuf` test above is: the corruption itself is probabilistic.
#[test]
fn every_element_of_a_struct_array_literal_is_retained() {
    let src = r#"
        struct Point { x: f64, y: f64 }
        fn main() -> i32 {
            let points: [Point; 3] = [
                Point(x: 0.0, y: 0.0),
                Point(x: 1.0, y: 1.0),
                Point(x: 2.0, y: 2.0)
            ];
            if points[0].x == 0.0 { 1 } else { 0 }
        }
        "#;
    let program = refcounted_cps(src);
    let retains = count_retains_for(&program, "Point");
    assert!(
        retains >= 3,
        "all three array elements must be retained before the array literal \
         releases their own local bindings -- got {retains} `Retain(Point)` calls"
    );
}

/// The same fix, verified end to end: reading every element back out of a
/// struct array literal, well after the array's own local element bindings
/// would otherwise have been released, must give back the real, correct
/// field values (not garbage from an already-reused allocation).
#[test]
fn a_struct_array_literals_elements_read_back_correctly_after_other_work() {
    let src = r#"
        struct Point { x: f64, y: f64 }
        fn scale(p: Point) -> f64 { p.x + p.y }
        fn main() -> i32 {
            let points: [Point; 3] = [
                Point(x: 1.0, y: 2.0),
                Point(x: 3.0, y: 4.0),
                Point(x: 5.0, y: 6.0)
            ];
            let mut total: f64 = 0.0;
            for i in 0..3 {
                total = total + scale(points[i]);
            };
            if total == 21.0 { 1 } else { 0 }
        }
        "#;
    assert_eq!(run_i32(src), 1);
}
