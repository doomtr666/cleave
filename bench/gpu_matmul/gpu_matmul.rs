// The device time of the matmul kernels cleave generates for the Vulkan
// backend, on nanoLM's own product shapes at its real size
// (`bench/gpu_matmul/matmul.cleave`), against the device's FMA ceiling as
// cleave compiles for it (`bench/gpu_matmul/fma.cleave`): the GPU's
// `bench/sgemm_shapes` and `bench/fma_roofline`. Times are the kernels' own
// (Vulkan timestamps, `CLEAVE_VK_PROFILE`), without transfers or the host.
//
//   rustc --edition 2024 -O bench/gpu_matmul/gpu_matmul.rs -o target/gpu_matmul.exe
//   target/gpu_matmul.exe [path/to/cleave.exe]      (from the repository root)
use std::collections::HashMap;
use std::process::Command;

/// How many times `matmul.cleave` calls each shape (its `REPS`).
const REPS: f64 = 20.0;
/// `fma.cleave`'s flops per call (2 * 8 accumulators * `STEPS` * `N`), and
/// its calls (`REPS`).
const FMA_FLOPS: f64 = 2.0 * 8.0 * 4096.0 * 1048576.0;
const FMA_REPS: f64 = 50.0;

// (function in `matmul.cleave`, m, n, k): the product of an m x k by a k x n.
const SHAPES: &[(&str, f64, f64, f64)] = &[
    ("qkvo_fwd", 1024.0, 768.0, 768.0),
    ("gate_up", 1024.0, 2048.0, 768.0),
    ("down", 1024.0, 768.0, 2048.0),
    ("head", 1024.0, 4096.0, 768.0),
    ("head_dx", 1024.0, 768.0, 4096.0),
    ("head_dw", 4096.0, 768.0, 1024.0),
    ("dw_qkvo", 768.0, 768.0, 1024.0),
    ("dw_up", 768.0, 2048.0, 1024.0),
    ("dx_up", 1024.0, 768.0, 2048.0),
    ("attn_qk", 1024.0, 1024.0, 64.0),
    ("attn_pv", 1024.0, 64.0, 1024.0),
    ("muon_xxt", 768.0, 768.0, 768.0),
    ("muon_wide", 768.0, 768.0, 2048.0),
];

/// Device microseconds per call of each function, summed over its kernels,
/// from the profile `cleave` prints running `program`: each kernel's line is
/// `<total> ms <share>%  <dispatches> x <mean> us  <function>_kernel #<id> [grid]`.
fn device_times(cleave: &str, program: &str, calls: f64) -> HashMap<String, f64> {
    let output = Command::new(cleave)
        .args(["--backend", "vulkan", "--no-openmp", "--no-debug-info", "--run", program])
        .env("CLEAVE_VK_PROFILE", "1")
        .output()
        .unwrap_or_else(|e| panic!("cannot run {cleave}: {e}"));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        String::from_utf8_lossy(&output.stdout).contains("main returned: 0"),
        "{program} failed:\n{stderr}"
    );
    let mut times = HashMap::new();
    for line in stderr.lines() {
        let words: Vec<&str> = line.split_whitespace().collect();
        // ["598.1", "ms", "29.5%", "20", "x", "29907.1", "us", "head_kernel", "#7", ...]
        if words.len() < 8 || words[1] != "ms" || words[4] != "x" || words[6] != "us" {
            continue;
        }
        let (Ok(dispatches), Ok(mean)) = (words[3].parse::<f64>(), words[5].parse::<f64>()) else { continue };
        let Some(function) = words[7].strip_suffix("_kernel") else { continue };
        *times.entry(function.to_string()).or_insert(0.0) += mean * dispatches / calls;
    }
    times
}

fn main() {
    let cleave = std::env::args().nth(1).unwrap_or_else(|| "target/release/cleave.exe".into());
    // The products first, the ceiling then, on a device already at its
    // clocks: measured cold (a few short kernels), it came out at half.
    let times = device_times(&cleave, "bench/gpu_matmul/matmul.cleave", REPS);
    let fma = device_times(&cleave, "bench/gpu_matmul/fma.cleave", FMA_REPS);
    let peak = FMA_FLOPS / (fma["fma_peak"] * 1e-6) / 1e9;
    println!("FMA ceiling: {peak:.0} GFLOP/s");
    println!("{:<12} {:>16} {:>10} {:>10} {:>7}", "shape", "m x n x k", "us/call", "GFLOP/s", "% peak");
    let mut total_flops = 0.0;
    let mut total_us = 0.0;
    for &(name, m, n, k) in SHAPES {
        let us = times.get(name).copied().unwrap_or(f64::NAN);
        let flops = 2.0 * m * n * k;
        let gflops = flops / (us * 1e-6) / 1e9;
        total_flops += flops;
        total_us += us;
        println!(
            "{name:<12} {:>16} {us:>10.1} {gflops:>10.0} {:>6.1}%",
            format!("{m}x{n}x{k}"),
            gflops / peak * 100.0
        );
    }
    let gflops = total_flops / (total_us * 1e-6) / 1e9;
    println!("{:<12} {:>16} {total_us:>10.1} {gflops:>10.0} {:>6.1}%", "all", "", gflops / peak * 100.0);
}
