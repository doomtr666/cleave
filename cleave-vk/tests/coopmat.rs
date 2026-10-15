//! Tensor cores through Vulkan's cooperative matrices (`doc/plan-quantization.md`, phase 0): a
//! hand-written bf16 fragment product, on every device that lists it, the validation layer on.

use cleave_mlir::Context;
use cleave_mlir::dialect::DialectRegistry;
use cleave_mlir::ir::Module;
use cleave_mlir::utility::register_all_dialects;
use cleave_vk::{Gpu, Options};

fn mma_bf16_spirv() -> Vec<u32> {
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    let context = Context::new();
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
    let module = Module::parse(&context, include_str!("../../cleave-mlir/tests/data/mma_bf16.mlir"))
        .expect("failed to parse the kernel");
    // SAFETY: `module` is a valid module, owned here.
    unsafe { cleave_mlir::serialize_spirv(module.to_raw(), "") }.expect("serialization failed")
}

/// A float32 holding a small integer, as the bfloat16 bits that represent it exactly (the top half
/// of the float32).
fn bf16_bits(x: f32) -> u16 {
    (x.to_bits() >> 16) as u16
}

/// `C = A B + C` on a 16x16x16 fragment, A and B in bfloat16, accumulated in float32, on each
/// device whose cooperative matrices include that product; small integers, so exact on every
/// path. Devices without it are skipped, saying so.
#[test]
fn a_bf16_fragment_product_runs_on_tensor_cores() {
    let devices = match cleave_vk::devices() {
        Ok(devices) if !devices.is_empty() => devices,
        other => {
            eprintln!("skipped: no Vulkan device ({other:?})");
            return;
        }
    };
    let spirv = mma_bf16_spirv();
    const N: usize = 16;
    let a: Vec<f32> = (0..N * N).map(|i| ((i / N + i % N) % 5) as f32 - 1.0).collect();
    let b: Vec<f32> = (0..N * N).map(|i| (((i / N) * (i % N)) % 3) as f32).collect();
    let c0: Vec<f32> = (0..N * N).map(|i| (i % 7) as f32).collect();
    let mut expected = c0.clone();
    for i in 0..N {
        for j in 0..N {
            for k in 0..N {
                expected[i * N + j] += a[i * N + k] * b[k * N + j];
            }
        }
    }
    for (index, info) in devices.iter().enumerate() {
        let supported = info.cooperative_matrices.iter().any(|s| {
            (s.m, s.n, s.k) == (16, 16, 16) && s.a == "bf16" && s.b == "bf16" && s.c == "f32" && s.scope == "subgroup"
        });
        if !supported {
            eprintln!("skipped on {}: no 16x16x16 bf16 cooperative-matrix product", info.name);
            continue;
        }
        let gpu = Gpu::open(&Options { device: Some(index.to_string()), validation: true })
            .unwrap_or_else(|e| panic!("opening device {index}: {e}"));
        let name = gpu.info().name.clone();
        let kernel = gpu.kernel(&spirv, "mma", 3).unwrap_or_else(|e| panic!("{name}: {e}"));
        let mut buffer_a = gpu.buffer(N * N * 2).unwrap();
        let mut buffer_b = gpu.buffer(N * N * 2).unwrap();
        let mut buffer_c = gpu.buffer(N * N * 4).unwrap();
        buffer_a.write(&a.iter().map(|&x| bf16_bits(x)).collect::<Vec<u16>>());
        buffer_b.write(&b.iter().map(|&x| bf16_bits(x)).collect::<Vec<u16>>());
        buffer_c.write(&c0);
        gpu.dispatch(&kernel, &[&buffer_a, &buffer_b, &buffer_c], [1, 1, 1]).unwrap_or_else(|e| panic!("{name}: {e}"));
        let mut c = vec![0f32; N * N];
        buffer_c.read(&mut c);
        assert_eq!(c, expected, "{name}");
        drop((kernel, buffer_a, buffer_b, buffer_c));
        let messages = gpu.validation_messages();
        assert!(messages.is_empty(), "{name}: validation messages:\n{}", messages.join("\n"));
        eprintln!("ok on {name}");
    }
}

fn peak_spirv() -> Vec<u32> {
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    let context = Context::new();
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
    let module = Module::parse(&context, include_str!("../../cleave-mlir/tests/data/mma_bf16_peak.mlir"))
        .expect("failed to parse the kernel");
    // SAFETY: `module` is a valid module, owned here.
    unsafe { cleave_mlir::serialize_spirv(module.to_raw(), "") }.expect("serialization failed")
}

/// The tensor cores' bf16 throughput with float32 accumulation (`mma_bf16_peak.mlir`), on each
/// device that has the product: the best of a few large dispatches, in TFLOPS. A measurement, not a
/// check: `cargo test --release -p cleave-vk --test coopmat -- --ignored --nocapture`.
#[test]
#[ignore]
fn bf16_tensor_core_throughput() {
    let Ok(devices) = cleave_vk::devices() else { return };
    let spirv = peak_spirv();
    const GROUPS: u32 = 8192;
    const SUBGROUPS: f64 = 4.0;
    const ITERS: f64 = 2048.0;
    const PRODUCTS: f64 = 4.0;
    let flops = f64::from(GROUPS) * SUBGROUPS * ITERS * PRODUCTS * 2.0 * 16.0 * 16.0 * 16.0;
    for (index, info) in devices.iter().enumerate() {
        if !info.cooperative_matrices.iter().any(|s| (s.m, s.n, s.k) == (16, 16, 16) && s.a == "bf16" && s.c == "f32") {
            eprintln!("skipped on {}: no 16x16x16 bf16 cooperative-matrix product", info.name);
            continue;
        }
        let gpu = Gpu::open(&Options { device: Some(index.to_string()), validation: false })
            .unwrap_or_else(|e| panic!("opening device {index}: {e}"));
        let name = gpu.info().name.clone();
        let kernel = gpu.kernel(&spirv, "peak", 3).unwrap_or_else(|e| panic!("{name}: {e}"));
        let statistics: Vec<String> = gpu.kernel_statistics(&kernel).iter().map(|(k, v)| format!("{k}={v}")).collect();
        eprintln!("{name}: {}", statistics.join(", "));
        let mut buffer_a = gpu.buffer(16 * 16 * 2).unwrap();
        let mut buffer_b = gpu.buffer(16 * 16 * 2).unwrap();
        let mut buffer_c = gpu.buffer(16 * 16 * 4).unwrap();
        buffer_a.write(&vec![0x3c00u16; 256]);
        buffer_b.write(&vec![0x3c00u16; 256]);
        buffer_c.write(&vec![0f32; 256]);
        let mut best = f64::MAX;
        for run in 0..6 {
            let start = std::time::Instant::now();
            gpu.dispatch(&kernel, &[&buffer_a, &buffer_b, &buffer_c], [GROUPS, 1, 1]).unwrap_or_else(|e| panic!("{name}: {e}"));
            let seconds = start.elapsed().as_secs_f64();
            if run > 0 {
                best = best.min(seconds);
            }
        }
        eprintln!("{name}: bf16 x bf16 + f32, {:.1} TFLOPS ({:.2} ms for {:.2} TFLOP)", flops / best / 1e12, best * 1e3, flops / 1e12);
    }
}

/// A tiled bf16 GEMM given as SPIR-V by `CLEAVE_COOPMAT_SPV` (a probe written outside the compiler,
/// `doc/plan-quantization.md`, phase 0), `C[M,N] = A[M,K] B[K,N]`, M = N = K = `CLEAVE_COOPMAT_SIZE`
/// (4096 by default), workgroups of `CLEAVE_COOPMAT_TILE` (128) square: sampled entries checked
/// exactly (small integers), the best of a few runs in TFLOPS. A measurement:
/// `CLEAVE_COOPMAT_SPV=gemm.spv cargo test --release -p cleave-vk --test coopmat tiled -- --ignored --nocapture`.
#[test]
#[ignore]
fn tiled_bf16_gemm_probe() {
    let Ok(path) = std::env::var("CLEAVE_COOPMAT_SPV") else {
        eprintln!("skipped: CLEAVE_COOPMAT_SPV not set");
        return;
    };
    let env = |name: &str, default: usize| std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default);
    let n = env("CLEAVE_COOPMAT_SIZE", 4096);
    let tile = env("CLEAVE_COOPMAT_TILE", 128);
    // A workgroup's rows and columns of `C`, when not square.
    let (tile_m, tile_n) = (env("CLEAVE_COOPMAT_TILE_M", tile), env("CLEAVE_COOPMAT_TILE_N", tile));
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let spirv: Vec<u32> = bytes.chunks_exact(4).map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]])).collect();
    let Ok(devices) = cleave_vk::devices() else { return };
    // Small integers in -2..=2, exact in bf16, every partial sum exact in f32.
    let value = |i: usize, salt: usize| (((i.wrapping_mul(2654435761) ^ salt) >> 7) % 5) as f32 - 2.0;
    let a: Vec<f32> = (0..n * n).map(|i| value(i, 0x9e37)).collect();
    let b: Vec<f32> = (0..n * n).map(|i| value(i, 0x85eb)).collect();
    let flops = 2.0 * (n as f64).powi(3);
    for (index, info) in devices.iter().enumerate() {
        if !info.cooperative_matrices.iter().any(|s| (s.m, s.n, s.k) == (16, 16, 16) && s.a == "bf16" && s.c == "f32") {
            eprintln!("skipped on {}: no 16x16x16 bf16 cooperative-matrix product", info.name);
            continue;
        }
        let gpu = Gpu::open(&Options { device: Some(index.to_string()), validation: false })
            .unwrap_or_else(|e| panic!("opening device {index}: {e}"));
        let name = gpu.info().name.clone();
        let kernel = gpu.kernel(&spirv, "main", 3).unwrap_or_else(|e| panic!("{name}: {e}"));
        let statistics: Vec<String> = gpu.kernel_statistics(&kernel).iter().map(|(k, v)| format!("{k}={v}")).collect();
        eprintln!("{name}: {}", statistics.join(", "));
        // In the device's own memory: a mapped `Gpu::buffer` may be system memory, read across PCIe.
        let buffer_a = gpu.device_buffer(n * n * 2).unwrap();
        let buffer_b = gpu.device_buffer(n * n * 2).unwrap();
        let buffer_c = gpu.device_buffer(n * n * 4).unwrap();
        let mut staging = gpu.buffer(n * n * 4).unwrap();
        staging.write(&a.iter().map(|&x| bf16_bits(x)).collect::<Vec<u16>>());
        gpu.copy(&staging, &buffer_a, n * n * 2).unwrap();
        staging.write(&b.iter().map(|&x| bf16_bits(x)).collect::<Vec<u16>>());
        gpu.copy(&staging, &buffer_b, n * n * 2).unwrap();
        let groups = [(n / tile_n) as u32, (n / tile_m) as u32, 1];
        let mut best = f64::MAX;
        for run in 0..6 {
            let start = std::time::Instant::now();
            gpu.dispatch(&kernel, &[&buffer_a, &buffer_b, &buffer_c], groups).unwrap_or_else(|e| panic!("{name}: {e}"));
            if run > 0 {
                best = best.min(start.elapsed().as_secs_f64());
            }
        }
        let mut c = vec![0f32; n * n];
        gpu.copy(&buffer_c, &staging, n * n * 4).unwrap();
        staging.read(&mut c);
        // `CLEAVE_COOPMAT_NOCHECK`: a dissection variant that computes something else on purpose.
        for s in 0..if std::env::var_os("CLEAVE_COOPMAT_NOCHECK").is_some() { 0 } else { 64 } {
            let (i, j) = ((s * 7919) % n, (s * 104729 + 13) % n);
            let expected: f32 = (0..n).map(|k| a[i * n + k] * b[k * n + j]).sum();
            assert_eq!(c[i * n + j], expected, "{name}: c[{i}, {j}]");
        }
        eprintln!("{name}: {n}^3 bf16 GEMM, {:.1} TFLOPS ({:.2} ms)", flops / best / 1e12, best * 1e3);
    }
}
