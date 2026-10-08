// Single-thread `cblas_sgemm` throughput on nanoLM v2's own product shapes,
// as `cleave-rt` calls it (OpenBLAS pinned to one thread, cleave's tasks
// supplying the parallelism), against this machine's measured single-core
// roofline (`bench/fma_roofline`, 353 GFLOP/s). Says how much of a step's
// GEMM time a better kernel could win back before writing one.
//
// With `threads` > 1, that many threads run the same shape at once, each on
// its own operands, as cleave's tasks do: the per-thread rate under an
// all-core load (lower clocks, shared L3 and memory bandwidth), the ceiling
// a training step's GEMMs can actually reach.
//
//   rustc --edition 2024 -O -C target-cpu=native bench/sgemm_shapes/sgemm_shapes.rs -o target/sgemm_shapes.exe
//   target/sgemm_shapes.exe [threads] [path/to/openblas.dll]
use std::ffi::c_void;
use std::sync::Barrier;
use std::time::Instant;

const PEAK_GFLOPS: f64 = 353.34;
const ROW_MAJOR: i32 = 101;
const NO_TRANS: i32 = 111;
const TRANS: i32 = 112;

type Sgemm = unsafe extern "C" fn(
    i32, i32, i32, i32, i32, i32, f32, *const f32, i32, *const f32, i32, f32, *mut f32, i32,
);
type SetThreads = unsafe extern "C" fn(i32);

#[link(name = "kernel32")]
unsafe extern "system" {
    fn LoadLibraryW(name: *const u16) -> *mut c_void;
    fn GetProcAddress(module: *mut c_void, name: *const i8) -> *mut c_void;
}

// (label, trans_a, trans_b, m, n, k): rows of a micro-batch `R` = 1024
// (4 windows of 256), width 384, SwiGLU hidden 1024, vocabulary 4096,
// attention per (window, head) 256 x 256 over a head of 64.
const SHAPES: &[(&str, bool, bool, i32, i32, i32)] = &[
    ("qkvo fwd   R x D @ D x D", false, false, 1024, 384, 384),
    ("gate/up    R x D @ D x H", false, false, 1024, 1024, 384),
    ("down       R x H @ H x D", false, false, 1024, 384, 1024),
    ("head       R x D @ (V x D)^T", false, true, 1024, 4096, 384),
    ("head dX    R x V @ V x D", false, false, 1024, 384, 4096),
    ("head dW    (R x V)^T @ R x D", true, false, 4096, 384, 1024),
    ("dW qkvo    (R x D)^T @ R x D", true, false, 384, 384, 1024),
    ("dW up      (R x D)^T @ R x H", true, false, 384, 1024, 1024),
    ("dX up      R x H @ (D x H)^T", false, true, 1024, 384, 1024),
    ("attn QK^T  L x DH @ (L x DH)^T", false, true, 256, 256, 64),
    ("attn PV    L x L @ L x DH", false, false, 256, 64, 256),
    ("muon XX^T  D x D @ (D x D)^T", false, true, 384, 384, 384),
    ("muon wide  D x H @ (D x H)^T", false, true, 384, 384, 1024),
    // A tile of 128 rows of the products above, as `blas_tile_and_fuse` calls
    // them: B (the weights) packed again on every call.
    ("tile qkvo  128 x D @ D x D", false, false, 128, 384, 384),
    ("tile gate  128 x D @ D x H", false, false, 128, 1024, 384),
    ("tile down  128 x H @ H x D", false, false, 128, 384, 1024),
    ("tile dX up 128 x H @ (D x H)^T", false, true, 128, 384, 1024),
];

/// The best seconds per call of one thread over 5 timed batches of `batch`
/// calls, every thread starting each batch together.
fn time_shape(sgemm: Sgemm, shape: (bool, bool, i32, i32, i32), batch: u64, barrier: &Barrier) -> f64 {
    let (ta, tb, m, n, k) = shape;
    // Row-major: A is m x k (k x m when transposed), B k x n (n x k).
    let a = vec![0.5f32; (m * k) as usize];
    let b = vec![0.25f32; (k * n) as usize];
    let mut c = vec![0f32; (m * n) as usize];
    let lda = if ta { m } else { k };
    let ldb = if tb { k } else { n };
    let mut call = || unsafe {
        sgemm(
            ROW_MAJOR,
            if ta { TRANS } else { NO_TRANS },
            if tb { TRANS } else { NO_TRANS },
            m, n, k, 1.0, a.as_ptr(), lda, b.as_ptr(), ldb, 0.0, c.as_mut_ptr(), n,
        )
    };
    for _ in 0..batch {
        call();
    }
    let mut best = f64::MAX;
    for _ in 0..5 {
        barrier.wait();
        let t = Instant::now();
        for _ in 0..batch {
            call();
        }
        best = best.min(t.elapsed().as_secs_f64() / batch as f64);
    }
    std::hint::black_box(&c);
    best
}

fn main() {
    let threads: usize = std::env::args().nth(1).map_or(1, |t| t.parse().expect("threads: a number"));
    let path = std::env::args().nth(2).unwrap_or_else(|| "target/openblas/bin/openblas.dll".into());
    let wide: Vec<u16> = path.encode_utf16().chain(Some(0)).collect();
    let (sgemm, set_threads) = unsafe {
        let module = LoadLibraryW(wide.as_ptr());
        assert!(!module.is_null(), "cannot load {path}");
        let sgemm = GetProcAddress(module, c"cblas_sgemm".as_ptr());
        let set = GetProcAddress(module, c"openblas_set_num_threads".as_ptr());
        assert!(!sgemm.is_null() && !set.is_null(), "{path}: missing symbols");
        (
            std::mem::transmute::<*mut c_void, Sgemm>(sgemm),
            std::mem::transmute::<*mut c_void, SetThreads>(set),
        )
    };
    unsafe { set_threads(1) };

    println!(
        "{:<32} {:>9} {:>9} {:>7}  ({threads} thread{} at once, per thread)",
        "shape", "us/call", "GFLOP/s", "% peak",
        if threads == 1 { "" } else { "s" }
    );
    for &(label, ta, tb, m, n, k) in SHAPES {
        let flops = 2.0 * m as f64 * n as f64 * k as f64;
        // About 0.2 s a batch at the single-core rate.
        let batch = ((0.2 * PEAK_GFLOPS * 0.6e9 / flops) as u64).max(1);
        let barrier = Barrier::new(threads);
        let times: Vec<f64> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..threads)
                .map(|_| s.spawn(|| time_shape(sgemm, (ta, tb, m, n, k), batch, &barrier)))
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        // The slowest thread: what a step waiting on all of them sees.
        let worst = times.iter().cloned().fold(0.0, f64::max);
        let gflops = flops / worst / 1e9;
        println!("{label:<32} {:>9.1} {gflops:>9.1} {:>6.1}%", worst * 1e6, gflops / PEAK_GFLOPS * 100.0);
    }
}
