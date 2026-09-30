//! `doc/plan-blas-native.md` §7.2 — the differential oracle that was
//! missing the first time (that attempt's own unexplained precision gap:
//! §0's own "0.9341/0.9019/0.5837" numbers, never root-caused). Deliberately
//! its own small pipeline, like the project's other JIT-executed harnesses
//! (`cleave/tests/unify_alloc.rs`), for the same reason: a real check, not just
//! "the verifier didn't complain."
//!
//! No `insert_refcounting` here, matching `mlir_lower.rs::run_i32_from_cps`'s
//! own established precedent for a short-lived, single-invocation test —
//! harmless to leak a handful of small tensors once, at process exit.

use cleave::cps::{collect_mlir_types, collect_struct_schemas, collect_units, convert_program};
use cleave::driver::compile;
use cleave::mlir_lower::lower_program;
use cleave::pipeline::{CodegenOptions, check_type_errors, lower_to_llvm, strip_ciface_wrapper_debug_info};
use cleave::registry::Registry;
use melior::Context;
use melior::dialect::DialectRegistry;
use melior::ir::operation::OperationLike;
use melior::pass;
use melior::utility::{parse_pass_pipeline, register_all_dialects};

fn context() -> Context {
    let dialect_registry = DialectRegistry::new();
    register_all_dialects(&dialect_registry);
    let context = Context::new();
    context.append_dialect_registry(&dialect_registry);
    context.load_all_available_dialects();
    context
}

/// Compiles `src` all the way to a real JIT invocation of `main() -> f32`,
/// mirroring `mlir_lower.rs::run_i32_from_cps`'s own three-stage tensor
/// pipeline (elementwise-to-linalg, one-shot-bufferize with function-
/// boundary rewriting, then the ordinary llvm-dialect tail) exactly — see
/// that function's own doc comments for why each stage is a *separate*
/// `PassManager`, and why the `--convert-to-llvm`/`--finalize-memref-to-
/// llvm`/`--convert-to-llvm` sequence is repeated. Only `cleave_alloc_rc`
/// (real `Tensor` construction) and `cleave_blas_sgemm` (`stdlib/blas/
/// blas.cleave`'s own real extern) are registered — this file's own
/// sources never reach `io`/`dynarray`/refcounting at all.
fn run_f32(context: &Context, src: &str) -> f32 {
    let (result, _sources) = compile(vec![("test.cleave".to_string(), src.to_string())], &[]);
    let program = result.unwrap_or_else(|e| panic!("compile failed: {e:?}"));
    let registry = Registry::build(&program);
    if let Err(diags) = check_type_errors(&program, &registry) {
        panic!("type check failed: {diags:?}");
    }
    let units = collect_units(&program, &registry);
    let cps_program = convert_program(units, None);
    let mlir_types = collect_mlir_types(&program);
    let struct_schemas = collect_struct_schemas(&program);
    let mut module = lower_program(context, &cps_program, &mlir_types, struct_schemas);
    assert!(
        module.as_operation().verify(),
        "generated MLIR module failed verification"
    );

    let pass_manager = pass::PassManager::new(context);
    pass_manager.add_pass(pass::linalg::create_convert_elementwise_to_linalg_pass());
    pass_manager
        .run(&mut module)
        .expect("convert-elementwise-to-linalg must succeed");

    let pass_manager = pass::PassManager::new(context);
    pass::bufferization::register_one_shot_bufferize_pass();
    parse_pass_pipeline(
        pass_manager.as_operation_pass_manager(),
        "builtin.module(one-shot-bufferize{bufferize-function-boundaries=true})",
    )
    .expect("failed to parse the one-shot-bufferize pass pipeline");
    pass_manager
        .run(&mut module)
        .expect("one-shot-bufferize must succeed");

    let pass_manager = pass::PassManager::new(context);
    pass_manager.add_pass(pass::linalg::create_convert_linalg_to_loops_pass());
    pass_manager.add_pass(pass::conversion::create_scf_to_control_flow());
    pass_manager.add_pass(pass::transform::create_canonicalizer());
    pass_manager.add_pass(pass::memref::create_expand_strided_metadata_pass());
    pass_manager.add_pass(pass::conversion::create_lower_affine());
    pass_manager.add_pass(pass::transform::create_canonicalizer());
    pass_manager.add_pass(pass::conversion::create_to_llvm());
    pass_manager.add_pass(pass::conversion::create_finalize_mem_ref_to_llvm());
    pass_manager.add_pass(pass::conversion::create_to_llvm());
    pass_manager.add_pass(pass::conversion::create_reconcile_unrealized_casts());
    pass_manager
        .run(&mut module)
        .expect("lowering to the llvm dialect must succeed");
    strip_ciface_wrapper_debug_info(&context, module.as_operation_mut());

    // `openmp: false`, `target_cpu: "native"` — matching `examples/mnist-
    // interop/build.rs`'s own real config exactly, not `CodegenOptions::
    // default()`'s own `openmp: true, target_cpu: None`. The `target_cpu`
    // one turned out to be *the* real gap, found live: without it, native
    // `matmul`'s own vectorized `vector.contract`/`vfmadd` lowering has no
    // real AVX2/AVX-512 target to compile against at all (a generic
    // baseline x86-64 target, no modern SIMD) — crippling *only* the
    // native path, since BLAS is an externally-compiled library, already
    // built with its own real target features, entirely unaffected by
    // this cleave-level setting. Comparing BLAS against artificially-
    // crippled native codegen is not comparing what the real kernel
    // (which *does* set `target_cpu: "native"`) actually runs.
    let mut options = CodegenOptions::default();
    options.openmp = false;
    options.target_cpu = Some("native".to_string());
    cleave::options::set(options.clone());
    lower_to_llvm(context, &mut module, &options).unwrap_or_else(|e| panic!("lower_to_llvm failed: {e:?}"));

    let engine = melior::ExecutionEngine::new(&module, 2, &[], false, false);
    unsafe {
        engine.register_symbol("cleave_alloc_rc", cleave_rt::cleave_alloc_rc as *mut ());
        engine.register_symbol("cleave_blas_sgemm", cleave_rt::cleave_blas_sgemm as *mut ());
        engine.register_symbol("memrefCopy", cleave_rt::memrefCopy as *mut ());
        engine.register_symbol("rand_seed", cleave_rt::rand_seed as *mut ());
        engine.register_symbol("rand_uniform_f32", cleave_rt::rand_uniform_f32 as *mut ());
        engine.register_symbol("cleave_alloc_local", cleave_rt::cleave_alloc_local as *mut ());
        engine.register_symbol("cleave_region_enter", cleave_rt::cleave_region_enter as *mut ());
        engine.register_symbol("cleave_region_exit", cleave_rt::cleave_region_exit as *mut ());
        engine.register_symbol("cleave_alloc_pool", cleave_rt::cleave_alloc_pool as *mut ());
        engine.register_symbol("cleave_release_pool", cleave_rt::cleave_release_pool as *mut ());
        engine.register_symbol("cleave_release", cleave_rt::cleave_release as *mut ());
        engine.register_symbol("cleave_retain", cleave_rt::cleave_retain as *mut ());
        engine.register_symbol("cleave_release_void", cleave_rt::cleave_release_void as *mut ());
        engine.register_symbol("cleave_alloc", cleave_rt::cleave_alloc as *mut ());
    }
    let mut out: f32 = f32::NAN;
    unsafe {
        engine
            .invoke_packed("main", &mut [&mut out as *mut f32 as *mut ()])
            .expect("JIT invocation must succeed");
    }
    out
}

/// Absolute correctness, against a hand-computed expected value, no
/// `matmul` involved at all — proves `blas::sgemm` itself is right before
/// ever comparing it against anything else. `A = [[1,2],[3,4]]` (2x2),
/// `B = [[5,6],[7,8]]` (2x2) -> `A@B = [[19,22],[43,50]]`.
#[test]
fn sgemm_computes_the_right_value_against_a_hand_computed_matrix_product() {
    let context = context();
    let src = r#"
        use linalg;
        use blas;
        fn main() -> f32 {
            let a: Tensor<f32,2,2> = Tensor::<f32,2,2>(data: [[1.0,2.0],[3.0,4.0]]);
            let b: Tensor<f32,2,2> = Tensor::<f32,2,2>(data: [[5.0,6.0],[7.0,8.0]]);
            let c: Tensor<f32,2,2> = Tensor::<f32,2,2>(data: [[0.0,0.0],[0.0,0.0]]);
            let r = sgemm(false, false, 1.0, a, b, 0.0, c);
            r[1, 1]
        }
        "#;
    let value = run_f32(&context, src);
    assert_eq!(value, 50.0, "A@B[1,1] should be 50.0 (43,50 second row)");
}

/// The real oracle (`doc/plan-blas-native.md` §7.2): `blas::sgemm` against
/// `MatMul::matmul` on the *same* data, same shape non-trivial enough to
/// exercise real accumulation (4x4, not 2x2) -- the missing check that
/// would have caught the first attempt's own unexplained precision gap.
#[test]
fn sgemm_agrees_with_native_matmul_on_the_same_data() {
    let context = context();
    let src = r#"
        use linalg;
        use blas;
        fn main() -> f32 {
            let a: Tensor<f32,4,4> = Tensor::<f32,4,4>(data: [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0, 16.0]
            ]);
            let b: Tensor<f32,4,4> = Tensor::<f32,4,4>(data: [
                [16.0, 15.0, 14.0, 13.0],
                [12.0, 11.0, 10.0, 9.0],
                [8.0, 7.0, 6.0, 5.0],
                [4.0, 3.0, 2.0, 1.0]
            ]);
            let zero: Tensor<f32,4,4> = Tensor::<f32,4,4>(data: [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0]
            ]);
            let native = matmul(a, b);
            let blas = sgemm(false, false, 1.0, a, b, 0.0, zero);
            let d00 = native[0,0] - blas[0,0];
            let d11 = native[1,1] - blas[1,1];
            let d22 = native[2,2] - blas[2,2];
            let d33 = native[3,3] - blas[3,3];
            let a0 = if d00 < 0.0 { 0.0 - d00 } else { d00 };
            let a1 = if d11 < 0.0 { 0.0 - d11 } else { d11 };
            let a2 = if d22 < 0.0 { 0.0 - d22 } else { d22 };
            let a3 = if d33 < 0.0 { 0.0 - d33 } else { d33 };
            let m01 = if a0 > a1 { a0 } else { a1 };
            let m23 = if a2 > a3 { a2 } else { a3 };
            if m01 > m23 { m01 } else { m23 }
        }
        "#;
    let max_abs_diff = run_f32(&context, src);
    assert!(
        max_abs_diff < 1e-4,
        "blas::sgemm and matmul disagree by {max_abs_diff} on the diagonal -- the exact \
         unexplained gap doc/plan-blas-native.md §0 flagged, now with a real reproduction"
    );
}

/// Not a correctness test — a real, isolated timing comparison, asked for
/// directly, to answer "is `sgemm` itself slower than native `matmul` for
/// `examples/mnist-interop`'s own real `l1` shape (32x784 @ 784x512),
/// independent of everything else in that program's own training loop."
/// Fresh random data built *inside* the loop on every iteration (`uniform`,
/// a real, genuinely non-foldable extern call) -- deliberately, not `Ring::
/// zero()`: an LLVM optimizer at `opt_level: 2` (this file's own `run_f32`)
/// can and will hoist a loop-invariant call with no observable side effect
/// out of the loop entirely, collapsing "200 iterations" down to one real
/// call -- found live, this is exactly the failure mode a *first* version
/// of this benchmark (built from `Ring::zero()`-seeded, unchanging `a`/`b`)
/// would have silently hit. Construction cost is identical between the two
/// variants below (same loop, same `uniform` calls), so it cancels out of
/// the comparison -- only `matmul` vs `sgemm` (plus `sgemm`'s own `to_
/// buffer` conversions) differs.
fn matmul_bench_src(iters: i32) -> String {
    format!(
        r#"
        use linalg;
        use rand;
        fn main() -> f32 {{
            let mut acc: f32 = 0.0;
            for _ in 0..{iters} {{
                let mut abuf: [f32;32,784] = mlir::memref::alloc();
                for i in 0..32 {{
                    for j in 0..784 {{
                        abuf[i,j] = uniform(-1.0, 1.0);
                    }};
                }};
                let a: Tensor<f32,32,784> = mlir::bufferization::to_tensor(abuf, restrict: "unit");
                let mut bbuf: [f32;784,512] = mlir::memref::alloc();
                for i in 0..784 {{
                    for j in 0..512 {{
                        bbuf[i,j] = uniform(-1.0, 1.0);
                    }};
                }};
                let b: Tensor<f32,784,512> = mlir::bufferization::to_tensor(bbuf, restrict: "unit");
                let c = matmul(a, b);
                acc = acc + c[0,0];
            }};
            acc
        }}
        "#
    )
}

fn sgemm_bench_src(iters: i32) -> String {
    format!(
        r#"
        use linalg;
        use blas;
        use rand;
        fn main() -> f32 {{
            let mut acc: f32 = 0.0;
            for _ in 0..{iters} {{
                let mut abuf: [f32;32,784] = mlir::memref::alloc();
                for i in 0..32 {{
                    for j in 0..784 {{
                        abuf[i,j] = uniform(-1.0, 1.0);
                    }};
                }};
                let a: Tensor<f32,32,784> = mlir::bufferization::to_tensor(abuf, restrict: "unit");
                let mut bbuf: [f32;784,512] = mlir::memref::alloc();
                for i in 0..784 {{
                    for j in 0..512 {{
                        bbuf[i,j] = uniform(-1.0, 1.0);
                    }};
                }};
                let b: Tensor<f32,784,512> = mlir::bufferization::to_tensor(bbuf, restrict: "unit");
                let zero: Tensor<f32,32,512> = Ring::zero();
                let c = sgemm(false, false, 1.0, a, b, 0.0, zero);
                acc = acc + c[0,0];
            }};
            acc
        }}
        "#
    )
}

/// Prints both timings to stderr (`--nocapture`) rather than asserting
/// anything — this is a real measurement to *read*, not a pass/fail gate;
/// asserting "BLAS must be faster" would be exactly the kind of unmeasured
/// assumption `doc/plan-blas-native.md` §7.3 already got burned by once.
#[test]
fn sgemm_vs_native_matmul_timing_on_mnist_interops_l1_shape() {
    let iters = 200;
    let native_src = matmul_bench_src(iters);
    let blas_src = sgemm_bench_src(iters);

    // A *fresh* `Context` per `run_f32` call, deliberately -- `lower_to_
    // llvm`'s own transform-dialect matmul-vectorize script registers a
    // named symbol into whichever `Context` it runs against; a second
    // `lower_to_llvm` call on the *same*, already-used `Context` hits a
    // real "doubly defined symbol @match_matmul" error (found live,
    // writing this benchmark) -- every other test in this file only ever
    // calls `run_f32` once per `context()`, so this never mattered before.
    //
    // One warm-up call each, discarded -- JIT compilation itself (not the
    // loop body) dominates a *single* `run_f32` call otherwise, exactly
    // the one-time cost this benchmark isn't trying to measure.
    let _ = run_f32(&context(), &native_src);
    let native_start = std::time::Instant::now();
    let _ = run_f32(&context(), &native_src);
    let native_elapsed = native_start.elapsed();

    let _ = run_f32(&context(), &blas_src);
    let blas_start = std::time::Instant::now();
    let _ = run_f32(&context(), &blas_src);
    let blas_elapsed = blas_start.elapsed();

    eprintln!(
        "native matmul: {native_elapsed:?} for {iters} iterations ({:?}/iter)",
        native_elapsed / iters as u32
    );
    eprintln!(
        "blas sgemm:    {blas_elapsed:?} for {iters} iterations ({:?}/iter)",
        blas_elapsed / iters as u32
    );
}
