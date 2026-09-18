//! End-to-end execution tests for `doc/plan-affine-ownership.md`'s Stage 2
//! — routing a proven-never-aliased construction through `cleave_alloc_
//! pool`/`cleave_release_pool` (no header, no retain/release runtime call)
//! instead of `cleave_alloc_rc`/`cleave_release`. **On by default** as of
//! §11-§14 landing (`mlir_lower.rs::lower_program`'s own doc comment) —
//! `CLEAVE_NO_AFFINE_STRUCTS` is the opt-out, used by the handful of tests
//! below whose own point is specifically the header-based path. Registers
//! the new symbols itself (`pipeline.rs::emit_object`'s own registration is
//! the production path). Every other JIT test harness in this crate that
//! registers `cleave_alloc_rc` also now registers `cleave_alloc_pool`/
//! `cleave_release_pool` right alongside it — found necessary the moment
//! this mechanism became the default rather than an opt-in flag: three of
//! them (`mlir_lower.rs`, `refcount.rs`, `user_guide.rs`) hit a real
//! `STATUS_STACK_BUFFER_OVERRUN` from an unresolved JIT symbol the moment
//! any of their own struct-typed test programs ran, confirming this wasn't
//! a hypothetical gap.

use cleave::cps::{collect_mlir_types, collect_struct_schemas};
use cleave::driver::compile;
use cleave::egraph::optimize_program;
use cleave::mlir_lower::lower_program;
use cleave::pipeline::{Backend, CodegenOptions, check_type_errors, lower_to_llvm};
use cleave::refcount::insert_refcounting;
use cleave::registry::Registry;
use melior::Context;
use melior::dialect::DialectRegistry;
use melior::ir::operation::OperationLike;
use melior::utility::register_all_dialects;
use std::sync::{Mutex, OnceLock};

// `CLEAVE_AFFINE_STRUCTS`/`CLEAVE_TRACE_ALLOC_TYPES` etc. are read via
// `std::env::var` at compile time (inside `lower_program`), and `cargo
// test` runs every test in this binary as threads of one process sharing
// one environment -- `env::set_var` in one test racing another test's own
// `env::var` read is a real, direct hazard, not a hypothetical one. A
// single mutex serializes every test in this file that touches the flag,
// matching the discipline this whole plan has used for other env-gated
// compiler flags all session.
static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
fn lock_env() -> std::sync::MutexGuard<'static, ()> {
    ENV_LOCK.get_or_init(|| Mutex::new(())).lock().unwrap()
}

/// Backs `extern fn opaque_sink(b: Boxed) -> i32` in
/// `many_short_lived_affine_constructions_run_correctly` below — a struct
/// passed across an `extern` boundary is always the plain opaque-pointer
/// representation (`mlir_lower.rs::ty_to_mlir`'s own `Ty::Con`/`Ty::App`
/// arm: `extern_boundary_structs` forces the heavy, `!llvm.ptr` form), and
/// the pointer this receives always points at `Boxed`'s own *data*
/// directly — `v: i32` at offset 0 — regardless of whether the allocation
/// underneath is headered (`cleave_alloc_rc`) or headerless
/// (`cleave_alloc_pool`, this test's whole point): neither allocator ever
/// exposes its own header, if any, to code on the far side of a call.
unsafe extern "C" fn opaque_sink(ptr: *mut u8) -> i32 {
    unsafe { *(ptr as *mut i32) }
}

fn context() -> Context {
    let dialect_registry = DialectRegistry::new();
    register_all_dialects(&dialect_registry);
    let context = Context::new();
    context.append_dialect_registry(&dialect_registry);
    context.load_all_available_dialects();
    context
}

/// `doc/plan-affine-ownership.md` §11-§14's own pool/cascade mechanism is
/// on **by default** now (every confirmed crash root-caused and fixed this
/// session, re-verified 5× under `CLEAVE_DEBUG_POOL=1` each, plus real
/// correct runs on both `examples/mnist-interop` and `examples/digits-
/// interop`) — `run_i32_inner` alone already exercises it, no env var
/// needed. This wrapper survives only so existing call sites don't all
/// need renaming; new tests should just call `run_i32_inner` directly.
fn run_i32_with_affine_structs(src: &str) -> i32 {
    run_i32_inner(src)
}

/// The explicit opt-out (`CLEAVE_NO_AFFINE_STRUCTS`, `lower_program`'s own
/// doc comment) — for the handful of tests whose own point is specifically
/// the *header-based* path (a differential check, or a safety net that
/// predates and is independent of the pool mechanism entirely), now that
/// plain `run_i32_inner` no longer means that on its own.
fn run_i32_with_affine_structs_disabled(src: &str) -> i32 {
    let _guard = lock_env();
    unsafe {
        std::env::set_var("CLEAVE_NO_AFFINE_STRUCTS", "1");
    }
    let result = std::panic::catch_unwind(|| run_i32_inner(src));
    unsafe {
        std::env::remove_var("CLEAVE_NO_AFFINE_STRUCTS");
    }
    match result {
        Ok(v) => v,
        Err(e) => std::panic::resume_unwind(e),
    }
}

fn run_i32_inner(src: &str) -> i32 {
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
    let cps_program = cleave::cps::eliminate_dead_code(cps_program);
    let mlir_types = collect_mlir_types(&program);
    let struct_schemas = collect_struct_schemas(&program);
    let escaping = cleave::escape::escaping_struct_vars(&cps_program);
    let cps_program = insert_refcounting(cps_program, &struct_schemas, &mlir_types, &escaping);

    let context = context();
    melior::utility::register_all_llvm_translations(&context);
    let mlir_types2 = collect_mlir_types(&program);
    let struct_schemas2 = collect_struct_schemas(&program);
    let mut module = lower_program(&context, &cps_program, &mlir_types2, struct_schemas2);
    assert!(module.as_operation().verify(), "module failed verification");

    // Reuses the real, already-battle-tested pipeline stage
    // (`pipeline.rs::lower_to_llvm`) instead of hand-reconstructing its own
    // multi-stage pass sequence here -- deliberately, so this test can
    // never silently drift from what `--emit-exe`/`--run` actually do.
    let options = CodegenOptions {
        opt_level: 2,
        openmp: false,
        target_cpu: None,
        target_features: None,
        backend: Backend::Cpu,
        ..Default::default()
    };
    lower_to_llvm(&context, &mut module, &options).expect("lower_to_llvm failed");

    let engine = melior::ExecutionEngine::new(&module, options.opt_level as usize, &[], true, false);
    unsafe {
        engine.register_symbol("cleave_alloc_rc", cleave_rt::cleave_alloc_rc as *mut ());
        engine.register_symbol("cleave_retain", cleave_rt::cleave_retain as *mut ());
        engine.register_symbol("cleave_release", cleave_rt::cleave_release as *mut ());
        engine.register_symbol(
            "cleave_release_void",
            cleave_rt::cleave_release_void as *mut (),
        );
        engine.register_symbol(
            "cleave_alloc_local",
            cleave_rt::cleave_alloc_local as *mut (),
        );
        engine.register_symbol(
            "cleave_region_enter",
            cleave_rt::cleave_region_enter as *mut (),
        );
        engine.register_symbol(
            "cleave_region_exit",
            cleave_rt::cleave_region_exit as *mut (),
        );
        engine.register_symbol("cleave_alloc_pool", cleave_rt::cleave_alloc_pool as *mut ());
        engine.register_symbol(
            "cleave_release_pool",
            cleave_rt::cleave_release_pool as *mut (),
        );
        engine.register_symbol("opaque_sink", opaque_sink as *mut ());

        let mut result: i32 = 0;
        engine
            .invoke_packed("main", &mut [&mut result as *mut i32 as *mut ()])
            .expect("JIT invocation failed");
        result
    }
}

/// The real `b = bump(b)` shape (`doc/backlog.md`'s array/loop-leak
/// entry), run to completion for real -- not just dumped -- with the
/// pool-allocator path active.
///
/// **Was `#[ignore]`d: confirmed `STATUS_HEAP_CORRUPTION`** —
/// `refcount::insert_refcounting`'s own `Release` targets the *loop-
/// carried parameter's* own `CVar` (`v469`-shaped), never a `PrimOp::
/// Struct`-bound one — but `affine_struct_vars` used to only classify
/// construction sites. Across a loop iteration, `v469` holds a pool-
/// allocated value (`bump`'s own result) on some iterations and a headered
/// one (the initial `b`) on the first — the *same* release call site can't
/// know which allocator backed *this* iteration's value, so it always
/// called the ordinary, header-reading `cleave_release`, corrupting
/// memory the iterations it was actually headerless.
///
/// **Fixed, in two parts (`doc/plan-affine-ownership.md` §11.4)**: (1)
/// `alias_analysis::analyze` no longer treats a call to a local `Fix`-
/// label (a loop, an `if`-join) as automatically aliased — every named
/// `CFunDef`, local or top-level, is now its own analyzed unit, so a
/// loop's own *entry* argument (`v468` here) can finally be proven never
/// aliased, exactly like any other construction, instead of being
/// unconditionally excluded by an analysis that had simply never been
/// taught to look inside a local label's own body; (2) `affine_struct_
/// vars` now also classifies a loop/if-join's own *carried* parameter
/// (`v469`) directly, eligible once *every* source that can ever feed
/// it — the entry argument and every back-edge, wherever they sit — is
/// itself already affine. Re-verified 5 consecutive runs under `CLEAVE_
/// DEBUG_POOL=1`: zero warnings of any kind, not even the harmless
/// `cleave_release_pool`-vs-`parked_insert` diagnostic gap `many_short_
/// lived_affine_constructions_run_correctly` above still shows — this one
/// no longer mixes `cleave_alloc_rc`/`cleave_alloc_pool` for the same
/// value at all, so there's nothing left to trip that gap either.
///
/// `0..2000`, not the original `0..2` this bug was first isolated at — the
/// e-graph has since gotten strong enough to fully constant-fold the
/// `0..2` case at compile time (confirmed directly: `CLEAVE_TRACE_AFFINE_
/// STRUCTS` reports zero candidate vars for that bound, and `--dump-cps-
/// optimized` shows the whole function reduced to a single literal
/// return), which had turned this exact test into an accidental false
/// negative — it kept reporting `ok` for a program that no longer
/// contained a real loop at all to exercise the bug. `0..2000` still
/// produces a genuine runtime loop (confirmed the same way).
#[test]
fn bump_shaped_loop_runs_correctly_through_the_pool_allocator() {
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
    assert_eq!(run_i32_with_affine_structs(src), 2000);
}

/// A struct with a field of its own value used in real arithmetic, still
/// affine, run through several independent local constructions (not
/// loop-carried) inside one loop -- confirms the pool allocator round-
/// trips correctly under real reuse pressure (many alloc/release cycles
/// through the same size class), not just a single value's own lifetime.
///
/// **Was `#[ignore]`d under an incorrect diagnosis** — `bump` here has
/// exactly one call site, entirely confined to a single loop iteration
/// (never fed to the loop's own back-edge, unlike the test above), so
/// `region_analysis::find_region_local_functions` marks it region-local:
/// its own construction is arena-backed (`cleave_alloc_local`), never
/// pool-backed at all, no matter what `alias_analysis::affine_struct_vars`
/// says about it in isolation. Two real, separate things needed fixing
/// together: (1) `affine_struct_vars` had no way to route a real call's
/// own resumption parameter through the pool even when its callee always
/// returns an already-affine value — `refcount::insert_refcounting`
/// releases that resumption parameter *directly* whenever the call's
/// result isn't threaded any further, so it needs the same treatment as a
/// construction site; (2) once that propagation existed, it had to be
/// taught to skip any function `region_analysis` already claimed for the
/// arena, or it would make a resumption parameter pool-eligible for a
/// value that was never pool-allocated to begin with. Both landed in
/// `alias_analysis::affine_struct_vars`. Re-verified directly, not just by
/// re-running this test: `CLEAVE_DEBUG_POOL=1` reports no double-release/
/// double-park signal at all — the one message it *does* still print
/// (`popped block ... was not marked parked`) is a separate, pre-existing,
/// harmless diagnostic gap (`cleave_release_pool` never calls this debug
/// build's own `parked_insert`, so any block legitimately handed back and
/// forth between `cleave_alloc_rc` and the pool — a real, already-tested
/// feature, `cleave_rt::rc_tests::pool_and_rc_allocations_freely_
/// interchange_the_same_size_class_blocks` — trips it on pop regardless of
/// correctness).
#[test]
fn many_short_lived_affine_constructions_run_correctly() {
    let src = "
        struct Boxed { v: i32, tag: [i32; 1] }
        fn bump(a: Boxed) -> Boxed { Boxed(v: a.v + 1, tag: [a.tag[0]]) }
        extern fn opaque_sink(b: Boxed) -> i32;
        fn main() -> i32 {
            let mut acc = 0;
            for i in 0..2000 {
                let a = Boxed(v: i, tag: [0]);
                acc = acc + opaque_sink(bump(a));
            };
            acc
        }
        ";
    // sum_{i=0}^{1999} (i + 1) = sum_{i=1}^{2000} i = 2000*2001/2
    assert_eq!(run_i32_with_affine_structs(src), 2000 * 2001 / 2);
}

/// A real, pool-mechanism-*independent* regression guard for a scare this
/// session ran into and ruled out, not the Stage 2 pool question at all:
/// `bump` here has exactly one call site, entirely confined to one loop
/// iteration, so `region_analysis::find_region_local_functions` marks it
/// region-local *unconditionally* — no gate involved — and its own
/// construction is arena-backed (`cleave_alloc_local`). The caller still
/// explicitly releases the renamed result (`refcount.rs` always emits a
/// release for a real call's own resumption parameter that isn't threaded
/// any further), which looked, from the CPS/MLIR alone, like it would call
/// the ordinary, header-reading `cleave_release` on a pointer that was
/// never `cleave_alloc_rc`-backed. It doesn't: `cleave_release`'s own
/// `is_in_arena` check (`cleave-rt/src/lib.rs`) already detects exactly
/// this case and skips the free-list entirely — this test exists to keep
/// that safety net honest under real, sustained reuse pressure (enough
/// alloc/release cycles to actually exercise it, not a handful),
/// independent of anything Stage 2 does — run with the pool mechanism
/// explicitly disabled, since that's the specific interaction (region-
/// local arena vs. the *ordinary* header release) this test is about, now
/// that plain `run_i32_inner` no longer implies that on its own.
#[test]
fn region_local_result_released_by_caller_with_the_pool_mechanism_disabled() {
    let src = "
        struct Boxed { v: i32, tag: [i32; 1] }
        fn bump(a: Boxed) -> Boxed { Boxed(v: a.v + 1, tag: [a.tag[0]]) }
        extern fn opaque_sink(b: Boxed) -> i32;
        fn main() -> i32 {
            let mut acc = 0;
            for i in 0..30000 {
                let a = Boxed(v: i, tag: [0]);
                acc = acc + opaque_sink(bump(a));
            };
            acc
        }
        ";
    // sum_{i=0}^{29999} (i + 1) = sum_{i=1}^{30000} i = 30000*30001/2
    assert_eq!(run_i32_with_affine_structs_disabled(src), 30000 * 30001 / 2);
}

/// `doc/plan-affine-ownership.md` §13/§14 — the real point of this whole
/// extension: `Outer` embeds a refcounted `Inner` field, never field-
/// mutated anywhere, so its own release now cascades *without* a header
/// all the way down, not just at the top level. Both `Outer` and `Inner`
/// are constructed directly in `main`'s own loop body (not behind a
/// region-local helper function, `region_local_result_released_by_caller_
/// with_the_gate_off`'s own doc comment has that trap) — `read_outer` is a
/// real, separately-compiled call, a pure borrow, so both constructions
/// survive as genuine `PrimOp::Struct` sites for this analysis to reason
/// about, exactly like `many_short_lived_affine_constructions_run_
/// correctly` above.
#[test]
fn a_nested_never_mutated_struct_cascades_through_the_pool_without_a_header() {
    let src = "
        struct Inner { v: i32, tag: [i32; 1] }
        struct Outer { inner: Inner, extra: i32 }
        fn read_outer(o: Outer) -> i32 { o.inner.v + o.extra }
        fn main() -> i32 {
            let mut acc = 0;
            for i in 0..2000 {
                let o = Outer(inner: Inner(v: i, tag: [0]), extra: i * 2);
                acc = acc + read_outer(o);
            };
            acc
        }
        ";
    // sum_{i=0}^{1999} (i + i*2) = 3 * sum_{i=0}^{1999} i = 3 * 1999*2000/2
    assert_eq!(run_i32_with_affine_structs(src), 3 * 1999 * 2000 / 2);
}

/// The exact same shape as `bump_shaped_loop_runs_correctly_through_the_
/// pool_allocator` above, this time with the pool mechanism explicitly
/// disabled (`CLEAVE_NO_AFFINE_STRUCTS=1`) -- confirms the ordinary,
/// header-based path still gives the identical result now that it's no
/// longer the default, so this test file also serves as a permanent
/// differential/fallback check between the two allocation strategies, not
/// just a standalone smoke test.
#[test]
fn the_same_program_gives_the_same_result_with_the_pool_mechanism_disabled() {
    let src = "
        struct Boxed { v: i32, tag: [i32; 1] }
        fn bump(a: Boxed) -> Boxed { Boxed(v: a.v + 1, tag: [a.tag[0]]) }
        fn main() -> i32 {
            let mut b = Boxed(v: 0, tag: [0]);
            for i in 0..1000 {
                b = bump(b);
            };
            b.v
        }
        ";
    assert_eq!(run_i32_with_affine_structs_disabled(src), 1000);
}
