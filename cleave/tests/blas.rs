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

use cleave::pipeline::CodegenOptions;
use cleave_mlir_shim::mlir::Context;
use cleave_mlir_shim::mlir::dialect::DialectRegistry;
use cleave_mlir_shim::mlir::utility::register_all_dialects;

fn context() -> Context {
    let dialect_registry = DialectRegistry::new();
    register_all_dialects(&dialect_registry);
    let context = Context::new();
    context.append_dialect_registry(&dialect_registry);
    context.load_all_available_dialects();
    context
}

/// Compiles and runs `src`'s `main` (`fn main() -> f32`) through the real
/// pipeline (`cleave::run`, what `--run` uses), `target_cpu: "native"` as
/// `examples/mnist-interop/build.rs` compiles: without it, the native
/// `matmul` has no AVX target and the comparison with BLAS (built for its
/// own target) is against crippled code. `main`'s four bytes read as an
/// `f32`; `_context` unused, kept for the call sites.
fn run_f32(_context: &Context, src: &str) -> f32 {
    let options = CodegenOptions {
        openmp: false,
        tasks: false,
        target_cpu: Some("native".to_string()),
        ..Default::default()
    };
    let bits = cleave::run::run_source("test.cleave", src, &options).unwrap_or_else(|e| panic!("{}", e.join("
")));
    f32::from_bits(bits as u32)
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
