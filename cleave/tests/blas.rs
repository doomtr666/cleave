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
use cleave_mlir::Context;
use cleave_mlir::dialect::DialectRegistry;
use cleave_mlir::utility::register_all_dialects;

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

/// A product of `a` (`m x k`, or `k x m` for `matmul_transpose_a`) and `b`
/// (`k x n`), `iters` times, as `call` writes it (`matmul(a2, b)`,
/// `matmul_transpose_a(a2, b)`, a `sgemm`). The operands are built once;
/// `a` is scaled a little every iteration, so the product can't be hoisted
/// out of the loop (it has no effect an optimizer must keep), for `m x k`
/// multiplications against the product's `m x k x n` FMAs.
fn product_bench_src(a_shape: (usize, usize), m: usize, n: usize, k: usize, call: &str, iters: i32) -> String {
    let (ar, ac) = a_shape;
    format!(
        r#"
        use linalg;
        use blas;
        use rand;
        fn main() -> f32 {{
            let mut abuf: [f32;{ar},{ac}] = mlir::memref::alloc();
            for i in 0..{ar} {{ for j in 0..{ac} {{ abuf[i,j] = uniform(-1.0, 1.0); }}; }};
            let mut a: Tensor<f32,{ar},{ac}> = mlir::bufferization::to_tensor(abuf, restrict: "unit");
            let mut bbuf: [f32;{k},{n}] = mlir::memref::alloc();
            for i in 0..{k} {{ for j in 0..{n} {{ bbuf[i,j] = uniform(-1.0, 1.0); }}; }};
            let b: Tensor<f32,{k},{n}> = mlir::bufferization::to_tensor(bbuf, restrict: "unit");
            let mut acc: f32 = 0.0;
            for _ in 0..{iters} {{
                a = Scale::scale(a, 0.999);
                let c: Tensor<f32,{m},{n}> = {call};
                acc = acc + c[0,0];
            }};
            acc
        }}
        "#
    )
}

/// Single-thread GFLOP/s of the native `matmul` (the `linalg` schedule) at
/// one count of FMAs, `M x K x N` = 32 x 784 x 512 (mnist-interop's `l1`),
/// for reductions from `K` = 32 to 3136, and of `matmul_transpose_a` on
/// `dW1`'s shape, `sgemm` for reference. A short reduction ran ~6x faster
/// per FMA than a long one (`doc/backlog.md`, "The `~6x` per-FLOP gap"): a
/// reduction chained through too few accumulators is latency-bound.
/// Compilation is excluded (two runs of different lengths, their
/// difference). A measurement to read, not a gate:
///
///   cargo test --release -p cleave --test blas -- --ignored --nocapture matmul_throughput
#[test]
#[ignore]
fn matmul_throughput_by_reduction_length() {
    // (label, A's shape, M, N, K, call)
    let cases: [(&str, (usize, usize), usize, usize, usize, &str); 6] = [
        ("matmul             M 784  K 32   N 512", (784, 32), 784, 512, 32, "matmul(a, b)"),
        ("matmul             M 196  K 128  N 512", (196, 128), 196, 512, 128, "matmul(a, b)"),
        ("matmul             M 32   K 784  N 512", (32, 784), 32, 512, 784, "matmul(a, b)"),
        ("matmul             M 8    K 3136 N 512", (8, 3136), 8, 512, 3136, "matmul(a, b)"),
        ("matmul_transpose_a M 784  K 32   N 512", (32, 784), 784, 512, 32, "matmul_transpose_a(a, b)"),
        (
            "sgemm              M 32   K 784  N 512",
            (32, 784),
            32,
            512,
            784,
            "sgemm(false, false, 1.0, a, b, 0.0, Ring::zero())",
        ),
    ];
    let (short, long) = (20, 220);
    for (label, a_shape, m, n, k, call) in cases {
        let fmas = (m * k * n) as f64;
        let time = |iters: i32| {
            let src = product_bench_src(a_shape, m, n, k, call, iters);
            let start = std::time::Instant::now();
            let _ = run_f32(&context(), &src);
            start.elapsed().as_secs_f64()
        };
        let _warm = time(short);
        let per_call = (time(long) - time(short)) / (long - short) as f64;
        eprintln!("{label}: {:7.1} us/call {:7.1} GFLOP/s", per_call * 1e6, 2.0 * fmas / per_call / 1e9);
    }
}
