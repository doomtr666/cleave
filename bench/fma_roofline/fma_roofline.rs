// Pure FMA-throughput microbenchmark -- no memory traffic in the hot loop,
// 16 independent accumulators (far more than needed to hide a ~4-cycle FMA
// latency across up to 2 issue ports, so this saturates real throughput,
// not latency) -- to measure this exact machine's real single-core
// AVX-512 FMA peak, rather than trust a spec-sheet assumption.
use std::arch::x86_64::*;
use std::time::Instant;

const LANES: f64 = 16.0; // f32 lanes per zmm (512 bits / 32 bits)
const ACCUMULATORS: usize = 16;
const FLOPS_PER_FMA: f64 = 2.0; // one multiply + one add

fn main() {
    if !is_x86_feature_detected!("avx512f") {
        println!("avx512f not available on this CPU -- cannot measure a 512-bit FMA peak");
        return;
    }
    unsafe { run() }
}

#[target_feature(enable = "avx512f")]
unsafe fn run() {
    // Converging fixed point (a<1, b>0 -> acc -> b/(1-a)), bounded, no
    // risk of drifting into inf/denormal territory over billions of
    // iterations -- keeps every FMA a normal, full-speed one throughout.
    let a = _mm512_set1_ps(0.999);
    let b = _mm512_set1_ps(0.001);
    let mut acc: [__m512; ACCUMULATORS] = [_mm512_set1_ps(1.0); ACCUMULATORS];

    let warmup_start = Instant::now();
    while warmup_start.elapsed().as_secs_f64() < 0.2 {
        for _ in 0..1000 {
            for j in 0..ACCUMULATORS {
                acc[j] = _mm512_fmadd_ps(acc[j], a, b);
            }
        }
    }

    let start = Instant::now();
    let mut batches: u64 = 0;
    const BATCH: u64 = 2000;
    while start.elapsed().as_secs_f64() < 2.0 {
        for _ in 0..BATCH {
            for j in 0..ACCUMULATORS {
                acc[j] = _mm512_fmadd_ps(acc[j], a, b);
            }
        }
        batches += BATCH;
        std::hint::black_box(&mut acc);
    }
    let elapsed = start.elapsed();

    let mut buf = [0.0f32; 16];
    let mut checksum = 0.0f64;
    for j in 0..ACCUMULATORS {
        _mm512_storeu_ps(buf.as_mut_ptr(), acc[j]);
        checksum += buf.iter().map(|x| *x as f64).sum::<f64>();
    }
    std::hint::black_box(checksum);

    let total_fma = batches as f64 * ACCUMULATORS as f64;
    let total_flops = total_fma * LANES * FLOPS_PER_FMA;
    let gflops = total_flops / elapsed.as_secs_f64() / 1e9;

    println!("checksum={checksum:.6} (sanity -- should be finite, near {})", 0.001 / (1.0 - 0.999));
    println!("elapsed={elapsed:?} batches={batches} accumulators={ACCUMULATORS}");
    println!("measured single-core AVX-512 FMA throughput: {gflops:.2} GFlop/s");
    println!();
    println!("for reference, at 4.5 GHz:");
    println!("  1x512-bit FMA port : {:.1} GFlop/s peak", 2.0 * 1.0 * 16.0 * 4.5);
    println!("  2x512-bit FMA port : {:.1} GFlop/s peak", 2.0 * 2.0 * 16.0 * 4.5);
}
