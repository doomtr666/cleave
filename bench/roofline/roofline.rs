// This machine's two roofline ceilings, with `threads` cores busy at once (as
// a training step keeps them): the AVX-512 FMA peak and the memory bandwidth
// at the L2, L3 and DRAM working-set sizes. A GEMM is judged against the
// first (`bench/sgemm_shapes`); an elementwise pass, a norm, an optimizer
// update, which move a few bytes per operation, against the second: their
// ceiling is `bytes moved / bandwidth`, not FMAs. With every core busy the
// chip may be power-bound, so the per-core figures can sit well below the
// single-core ones (`bench/fma_roofline`).
//
//   rustc --edition 2024 -O -C target-cpu=native bench/roofline/roofline.rs -o target/roofline.exe
//   target/roofline.exe [threads]          (default: 8)
use std::arch::x86_64::*;
use std::sync::Barrier;
use std::time::Instant;

const ACCUMULATORS: usize = 16;

/// FMAs per thread per second over ~1 s, every thread running at once.
#[target_feature(enable = "avx512f")]
unsafe fn fma_loop(seconds: f64) -> f64 {
    let a = _mm512_set1_ps(0.999);
    let b = _mm512_set1_ps(0.001);
    let mut acc = [_mm512_set1_ps(1.0); ACCUMULATORS];
    let start = Instant::now();
    let mut batches = 0u64;
    while start.elapsed().as_secs_f64() < seconds {
        for _ in 0..2000 {
            for x in acc.iter_mut() {
                *x = _mm512_fmadd_ps(*x, a, b);
            }
        }
        batches += 2000;
        std::hint::black_box(&mut acc);
    }
    batches as f64 * ACCUMULATORS as f64 * 16.0 * 2.0 / start.elapsed().as_secs_f64()
}

/// Bytes per second one thread moves through `kernel` on arrays of `len`
/// floats, best of a few repetitions, every thread starting together.
fn stream(kernel: &str, len: usize, barrier: &Barrier) -> f64 {
    let mut a = vec![1.0f32; len];
    let b = vec![2.0f32; len];
    let c = vec![3.0f32; len];
    // Read and written bytes per element (STREAM's convention: the write
    // allocate isn't counted).
    let bytes_per_element = match kernel {
        "copy" | "scale" => 8.0,
        _ => 12.0,
    };
    // About 0.1 s a repetition at 20 GB/s.
    let reps = ((2e9 / (len as f64 * bytes_per_element)) as usize).max(1);
    let mut best = f64::MAX;
    for _ in 0..5 {
        barrier.wait();
        let t = Instant::now();
        for _ in 0..reps {
            match kernel {
                "copy" => a.copy_from_slice(&b),
                "scale" => a.iter_mut().zip(&b).for_each(|(x, y)| *x = 3.0 * y),
                "add" => a.iter_mut().zip(b.iter().zip(&c)).for_each(|(x, (y, z))| *x = y + z),
                _ => a.iter_mut().zip(b.iter().zip(&c)).for_each(|(x, (y, z))| *x = y + 3.0 * z),
            }
            std::hint::black_box(&mut a);
        }
        best = best.min(t.elapsed().as_secs_f64() / reps as f64);
    }
    len as f64 * bytes_per_element / best
}

fn main() {
    let threads: usize = std::env::args().nth(1).map_or(8, |t| t.parse().expect("threads: a number"));
    assert!(is_x86_feature_detected!("avx512f"), "no AVX-512 here");

    let per_thread: Vec<f64> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads).map(|_| s.spawn(|| unsafe { fma_loop(1.0) })).collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let total: f64 = per_thread.iter().sum();
    let slowest = per_thread.iter().cloned().fold(f64::MAX, f64::min);
    println!(
        "FMA peak, {threads} thread(s): {:.0} GFLOP/s total, {:.0} per thread (slowest {:.0})",
        total / 1e9,
        total / threads as f64 / 1e9,
        slowest / 1e9
    );

    println!();
    println!("bandwidth, {threads} thread(s) at once, GB/s total (per thread)");
    println!("{:>14} {:>16} {:>16} {:>16} {:>16}", "per thread", "copy", "scale", "add", "triad");
    // Per thread: 3 arrays of `len` floats. 192 KiB fits a 1 MiB L2; 3 MiB
    // a thread's share of a 32 MiB L3; 192 MiB is DRAM.
    for (label, len) in [("192 KiB (L2)", 16 << 10), ("3 MiB (L3)", 256 << 10), ("192 MiB (DRAM)", 16 << 20)] {
        let mut row = format!("{label:>14}");
        for kernel in ["copy", "scale", "add", "triad"] {
            let barrier = Barrier::new(threads);
            let rates: Vec<f64> = std::thread::scope(|s| {
                let handles: Vec<_> = (0..threads).map(|_| s.spawn(|| stream(kernel, len, &barrier))).collect();
                handles.into_iter().map(|h| h.join().unwrap()).collect()
            });
            let total: f64 = rates.iter().sum();
            row += &format!(" {:>8.0} ({:>5.0})", total / 1e9, total / threads as f64 / 1e9);
        }
        println!("{row}");
    }
}
