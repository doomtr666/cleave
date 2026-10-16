//! The Vulkan backend (`doc/plan-vulkan.md`), through the CLI: programs
//! compiled with `--backend vulkan` and run on every Vulkan device of the
//! machine, the validation layer on. Skipped, saying so, without a device.

use std::process::Command;

/// Runs `src` with `--backend vulkan` on the device `index`, returning
/// stdout and stderr.
fn run_on(index: usize, name: &str, src: &str) -> (String, String) {
    let dir = std::env::temp_dir().join(format!("cleave-vulkan-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join(format!("{name}.cleave"));
    std::fs::write(&source, src).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_cleave"))
        .args(["--backend", "vulkan", "--no-openmp", "--no-debug-info", "--run"])
        .arg(&source)
        .env("CLEAVE_VK_DEVICE", index.to_string())
        .env("CLEAVE_VK_VALIDATION", "1")
        .output()
        .expect("cannot run cleave");
    (String::from_utf8_lossy(&output.stdout).into_owned(), String::from_utf8_lossy(&output.stderr).into_owned())
}

/// Runs `src` on the CPU, returning stdout: the reference a device's values are compared against.
fn run_cpu(name: &str, src: &str) -> String {
    let dir = std::env::temp_dir().join(format!("cleave-vulkan-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join(format!("{name}-cpu.cleave"));
    std::fs::write(&source, src).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_cleave"))
        .args(["--no-openmp", "--no-debug-info", "--run"])
        .arg(&source)
        .output()
        .expect("cannot run cleave");
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn devices() -> Vec<cleave_vk::DeviceInfo> {
    match cleave_vk::devices() {
        Ok(devices) if !devices.is_empty() => devices,
        other => {
            eprintln!("skipped: no Vulkan device ({other:?})");
            Vec::new()
        }
    }
}

/// An elementwise sum of two random 4x256 tensors, in a function of its own,
/// runs as one kernel on each device and matches the CPU's sum of the same
/// elements; the validation layer says nothing.
#[test]
fn an_elementwise_sum_runs_as_a_kernel_on_every_device() {
    let src = "
        use nn;
        #[no_inline]
        fn add(a: Tensor<f32, 4, 256>, b: Tensor<f32, 4, 256>) -> Tensor<f32, 4, 256> {
            a + b
        }
        fn main() -> i32 {
            rand_seed(1);
            let a: Tensor<f32, 4, 256> = Init::he();
            let b: Tensor<f32, 4, 256> = Init::he();
            let c = add(a, b);
            let mut wrong: i32 = 0;
            for i in 0..4 {
                for j in 0..256 {
                    if c[i, j] != a[i, j] + b[i, j] {
                        wrong = wrong + 1;
                    };
                };
            };
            wrong
        }
    ";
    for (index, device) in devices().iter().enumerate() {
        let (stdout, stderr) = run_on(index, "elementwise_sum", src);
        assert!(stdout.contains("main returned: 0"), "{}: stdout: {stdout}\nstderr: {stderr}", device.name);
        assert!(!stderr.contains("cleave Vulkan validation"), "{}: {stderr}", device.name);
        eprintln!("ok on {}", device.name);
    }
}

/// A matmul runs as kernels (the zero fill, then a thread per output
/// element: Phase 0's naive form) and matches the scalar sum of products.
#[test]
fn a_matmul_runs_as_kernels_on_every_device() {
    let src = "
        use nn;
        #[no_inline]
        fn product(a: Tensor<f32, 64, 32>, b: Tensor<f32, 32, 48>) -> Tensor<f32, 64, 48> {
            matmul(a, b)
        }
        fn main() -> i32 {
            rand_seed(1);
            let a: Tensor<f32, 64, 32> = Init::he();
            let b: Tensor<f32, 32, 48> = Init::he();
            let c = product(a, b);
            let mut wrong: i32 = 0;
            for i in 0..64 {
                for j in 0..48 {
                    let mut s: f32 = 0.0;
                    for k in 0..32 {
                        s = s + a[i, k] * b[k, j];
                    };
                    let d: f32 = c[i, j] - s;
                    if d > 0.0001 or d < -0.0001 {
                        wrong = wrong + 1;
                    };
                };
            };
            wrong
        }
    ";
    for (index, device) in devices().iter().enumerate() {
        let (stdout, stderr) = run_on(index, "matmul", src);
        assert!(stdout.contains("main returned: 0"), "{}: stdout: {stdout}\nstderr: {stderr}", device.name);
        assert!(!stderr.contains("cleave Vulkan validation"), "{}: {stderr}", device.name);
    }
}

/// Runs `src` like `run_on`, with the runtime's transfer counts.
fn stats_on(index: usize, name: &str, src: &str) -> (String, String) {
    let dir = std::env::temp_dir().join(format!("cleave-vulkan-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join(format!("{name}.cleave"));
    std::fs::write(&source, src).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_cleave"))
        .args(["--backend", "vulkan", "--no-openmp", "--no-debug-info", "--run"])
        .arg(&source)
        .env("CLEAVE_VK_DEVICE", index.to_string())
        .env("CLEAVE_VK_VALIDATION", "1")
        .env("CLEAVE_VK_STATS", "1")
        .output()
        .expect("cannot run cleave");
    (String::from_utf8_lossy(&output.stdout).into_owned(), String::from_utf8_lossy(&output.stderr).into_owned())
}

/// Tensors stay on the device across kernels (`doc/plan-vulkan.md`, Phases
/// 1 and 2): ten steps of `x = x + w` upload `x` and `w` once each (4 KiB each),
/// the loop's copy of `x` running as a kernel too, and download `x` once,
/// when the host reads it; the result matches.
#[test]
fn tensors_stay_on_the_device_across_kernels() {
    let src = "
        use nn;
        #[no_inline]
        fn step(x: Tensor<f32, 4, 256>, w: Tensor<f32, 4, 256>) -> Tensor<f32, 4, 256> {
            x + w
        }
        fn main() -> i32 {
            rand_seed(1);
            let w: Tensor<f32, 4, 256> = Init::he();
            let mut x: Tensor<f32, 4, 256> = Init::he();
            let x0 = x[2, 100];
            for i in 0..10 {
                x = step(x, w);
            };
            let d = x[2, 100] - (x0 + 10.0 * w[2, 100]);
            if d < 0.001 and d > -0.001 { 0 } else { 1 }
        }
    ";
    for (index, device) in devices().iter().enumerate() {
        let (stdout, stderr) = stats_on(index, "resident", src);
        assert!(stdout.contains("main returned: 0"), "{}: stdout: {stdout}\nstderr: {stderr}", device.name);
        assert!(!stderr.contains("cleave Vulkan validation"), "{}: {stderr}", device.name);
        assert!(stderr.contains("8192 bytes uploaded, 4096 bytes downloaded"), "{}: {stderr}", device.name);
        // Phase 2: the 21 kernels recorded into one batch, submitted once,
        // when the host reads `x`.
        assert!(stderr.contains("21 dispatches") && stderr.contains(", 1 submits"), "{}: {stderr}", device.name);
    }
}

/// A loop whose body calls a function kept out of line for its size
/// (`cleave-limit-inlining`) still runs as one kernel, the call inlined into
/// it: `rms_norm`'s rows call `row_rrms`, whose 256-wide reduction is
/// unrolled past the threshold at this width; the row loop stayed on the
/// host, a dispatch per row (nanoLM's small configuration: 600k of them a
/// step).
#[test]
fn a_loop_calling_an_out_of_line_function_is_one_kernel() {
    let src = "
        use nn;
        fn main() -> i32 {
            rand_seed(1);
            let x: Tensor<f32, 256, 256> = Init::he();
            let g: Tensor<f32, 1, 256> = Init::he();
            let y = RmsNorm::rms_norm(x, g);
            let mut s: f32 = 0.0;
            for c in 0..256 { s = s + x[7, c] * x[7, c]; };
            let expected = x[7, 9] / sqrt(s / 256.0 + 0.00001) * g[0, 9];
            let d = y[7, 9] - expected;
            if d < 0.0001 and d > -0.0001 { 0 } else { 1 }
        }
    ";
    for (index, device) in devices().iter().enumerate() {
        let (stdout, stderr) = stats_on(index, "out_of_line_call", src);
        assert!(stdout.contains("main returned: 0"), "{}: stdout: {stdout}
stderr: {stderr}", device.name);
        assert!(!stderr.contains("cleave Vulkan validation"), "{}: {stderr}", device.name);
        assert!(stderr.contains(" 1 dispatches"), "{}: {stderr}", device.name);
    }
}

/// Causal attention moves only what it must: its three inputs go up once
/// (its out-parameter, fresh, is discarded though the head loop carries
/// it), every head's scores and weights stay on the device, a
/// fresh allocation's undefined contents never uploaded (`causal_softmax`'s
/// weights, which its kernel reads back), and the host reads one element
/// back. Up: three 256 KiB tensors and 12 bytes of scalar kernel arguments.
/// Found as 512 KiB moved per head in nanoLM's step: the softmax loop
/// stayed on the host while `sgemm` still took raw pointers, and each fresh
/// buffer was uploaded. Its four heads are one dispatch per kernel
/// (`cleave-fold-loops-into-grids`), and its products start from the zeros
/// their results were filled with (`cleave-tile-matmuls-for-gpu`): four in
/// all.
#[test]
fn attention_moves_only_its_inputs() {
    let src = "
        use nn;
        fn main() -> i32 {
            rand_seed(1);
            let q: Tensor<f32, 256, 256> = Init::he();
            let k: Tensor<f32, 256, 256> = Init::he();
            let v: Tensor<f32, 256, 256> = Init::he();
            let o = CausalAttention::causal_attention(q, k, v, AttentionShape::<256, 64>());
            let d = o[0, 5] - v[0, 5];
            if d < 0.0001 and d > -0.0001 { 0 } else { 1 }
        }
    ";
    for (index, device) in devices().iter().enumerate() {
        let (stdout, stderr) = stats_on(index, "attention", src);
        assert!(stdout.contains("main returned: 0"), "{}: stdout: {stdout}
stderr: {stderr}", device.name);
        assert!(!stderr.contains("cleave Vulkan validation"), "{}: {stderr}", device.name);
        assert!(stderr.contains("786444 bytes uploaded") && stderr.contains(", 1 submits"), "{}: {stderr}", device.name);
        assert!(stderr.contains(": 4 dispatches"), "{}: {stderr}", device.name);
    }
}

/// A kernel's loop that yields a value and doesn't iterate gives its initial
/// value: thread 0's `for j in 1..1` here. MLIR's SPIR-V conversion carries
/// loop results in variables only the loop body stores, so it read
/// uninitialized memory (`cleave-guard-zero-trip-loops`): the first row of
/// `causal_softmax`, a NaN in the miniature transformer's gradients one run
/// in two.
#[test]
fn a_kernel_loop_that_does_not_iterate_yields_its_initial_value() {
    let src = "
        use nn;
        #[no_inline]
        fn running_max(x: Tensor<f32, 16, 16>) -> Tensor<f32, 16, 1> {
            let mut y: [[f32; 1]; 16] = mlir::memref::alloc();
            for i in 0..16 {
                let mut m = x[i, 0];
                for j in 1..i + 1 { if x[i, j] > m { m = x[i, j]; }; };
                y[i, 0] = m;
            };
            Tensor(data: y)
        }
        fn main() -> i32 {
            rand_seed(1);
            let x: Tensor<f32, 16, 16> = Init::he();
            let y = running_max(x);
            if y[0, 0] == x[0, 0] { 0 } else { 1 }
        }
    ";
    for (index, device) in devices().iter().enumerate() {
        let (stdout, stderr) = stats_on(index, "zero_trip", src);
        assert!(stdout.contains("main returned: 0"), "{}: stdout: {stdout}
stderr: {stderr}", device.name);
        assert!(!stderr.contains("cleave Vulkan validation"), "{}: {stderr}", device.name);
        assert!(stderr.contains(" 1 dispatches"), "{}: {stderr}", device.name);
    }
}

/// A loop summing per-row values (`sparse_cross_entropy`'s `total = total +
/// lse(z, b) - z[b, y[b]]`) runs its rows as a kernel, the sum itself
/// replayed on the host in the same order from the rows' values
/// (`splitCarriedChain`): the CPU's loss to the bit, and two 64-float
/// buffers downloaded instead of the 128 KiB of logits (nanoLM's loss read
/// back 1 GiB a step).
#[test]
fn a_loss_reads_back_its_rows_values_not_its_logits() {
    let src = "
        use nn;
        use io;
        #[no_inline]
        fn logits(x: Tensor<f32, 64, 512>, w: Tensor<f32, 64, 512>) -> Tensor<f32, 64, 512> { x * w }
        fn main() -> i32 {
            rand_seed(1);
            let x: Tensor<f32, 64, 512> = Init::he();
            let w: Tensor<f32, 64, 512> = Init::he();
            let y = [for i in 0..64: i * 7];
            println(sparse_cross_entropy(logits(x, w), y));
            0
        }
    ";
    let devices = devices();
    if devices.is_empty() {
        return;
    }
    let dir = std::env::temp_dir().join(format!("cleave-vulkan-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join("loss.cleave");
    std::fs::write(&source, src).unwrap();
    let cpu = Command::new(env!("CARGO_BIN_EXE_cleave"))
        .args(["--no-openmp", "--no-debug-info", "--run"])
        .arg(&source)
        .output()
        .expect("cannot run cleave");
    let cpu = String::from_utf8_lossy(&cpu.stdout).into_owned();
    let loss = cpu.lines().next().unwrap_or_default().to_string();
    assert!(loss.parse::<f32>().is_ok(), "cpu: {cpu}");
    for (index, device) in devices.iter().enumerate() {
        let (stdout, stderr) = stats_on(index, "loss", src);
        assert_eq!(stdout.lines().next(), Some(loss.as_str()), "{}: stdout: {stdout}
stderr: {stderr}", device.name);
        assert!(!stderr.contains("cleave Vulkan validation"), "{}: {stderr}", device.name);
        assert!(stderr.contains(" 512 bytes downloaded"), "{}: {stderr}", device.name);
    }
}

/// Products the device runs tiled (`cleave-tile-matmuls-for-gpu`: blocks of
/// the result per workgroup, slices of the operands through workgroup
/// memory) give the CPU's results in each layout: `matmul`,
/// `matmul_transpose_b`, `matmul_transpose_a`, and a shape only the smaller
/// tiling divides. One large enough to fill the device sums each element
/// over `k` in order, the CPU's result to the bit (elements compared: a sum
/// of them is regrouped on the device); the small ones make too
/// few workgroups and split `k` (two slices added after), equal within
/// rounding.
#[test]
fn tiled_products_match_the_cpu() {
    let src = "
        use nn;
        use io;
        #[no_inline]
        fn p1(a: Tensor<f32, 256, 64>, b: Tensor<f32, 64, 128>) -> Tensor<f32, 256, 128> { matmul(a, b) }
        #[no_inline]
        fn p2(a: Tensor<f32, 256, 64>, b: Tensor<f32, 128, 64>) -> Tensor<f32, 256, 128> { matmul_transpose_b(a, b) }
        #[no_inline]
        fn p3(a: Tensor<f32, 64, 256>, b: Tensor<f32, 64, 128>) -> Tensor<f32, 256, 128> { matmul_transpose_a(a, b) }
        #[no_inline]
        fn p4(a: Tensor<f32, 128, 64>, b: Tensor<f32, 64, 64>) -> Tensor<f32, 128, 64> { matmul(a, b) }
        #[no_inline]
        fn p5(a: Tensor<f32, 1024, 64>, b: Tensor<f32, 64, 2048>) -> Tensor<f32, 1024, 2048> { matmul(a, b) }
        fn checksum(c: Tensor<f32, 256, 128>) -> f32 {
            let mut s: f32 = 0.0;
            for i in 0..256 { for j in 0..128 { s = s + c[i, j] * (i * 7 - j * 3).to(); }; };
            s
        }
        fn main() -> i32 {
            rand_seed(3);
            let a: Tensor<f32, 256, 64> = Init::he();
            let at: Tensor<f32, 64, 256> = Init::he();
            let b: Tensor<f32, 64, 128> = Init::he();
            let bt: Tensor<f32, 128, 64> = Init::he();
            let x: Tensor<f32, 128, 64> = Init::he();
            let w: Tensor<f32, 64, 64> = Init::he();
            println(checksum(p1(a, b)));
            println(checksum(p2(a, bt)));
            println(checksum(p3(at, b)));
            let y = p4(x, w);
            let mut s: f32 = 0.0;
            for i in 0..128 { for j in 0..64 { s = s + y[i, j] * (i - j).to(); }; };
            println(s);
            let big_a: Tensor<f32, 1024, 64> = Init::he();
            let big_b: Tensor<f32, 64, 2048> = Init::he();
            let z = p5(big_a, big_b);
            println(z[0, 0]);
            println(z[517, 1033]);
            println(z[1023, 2047]);
            0
        }
    ";
    let devices = devices();
    if devices.is_empty() {
        return;
    }
    let dir = std::env::temp_dir().join(format!("cleave-vulkan-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join("tiled_products.cleave");
    std::fs::write(&source, src).unwrap();
    let cpu = Command::new(env!("CARGO_BIN_EXE_cleave"))
        .args(["--no-openmp", "--no-debug-info", "--run"])
        .arg(&source)
        .output()
        .expect("cannot run cleave");
    let cpu = String::from_utf8_lossy(&cpu.stdout).into_owned();
    assert!(cpu.contains("main returned: 0"), "cpu: {cpu}");
    let lines = |out: &str| -> Vec<String> { out.lines().map(str::to_string).collect() };
    let expected = lines(&cpu);
    // The compiler's own tilings, then one reading both operands from
    // device memory 4 elements at a time (`CLEAVE_GPU_MATMUL`'s `vec` 3: rows
    // contiguous as they come, `k` contiguous transposed in registers).
    for ((index, device), forced) in
        devices.iter().enumerate().flat_map(|d| [(d, None), (d, Some("64,64,16,8,8,0,3"))])
    {
        let mut command = Command::new(env!("CARGO_BIN_EXE_cleave"));
        command
            .args(["--backend", "vulkan", "--no-openmp", "--no-debug-info", "--run"])
            .arg(&source)
            .env("CLEAVE_VK_DEVICE", index.to_string())
            .env("CLEAVE_VK_VALIDATION", "1");
        if let Some(config) = forced {
            command.env("CLEAVE_GPU_MATMUL", config);
        }
        let output = command.output().expect("cannot run cleave");
        let (stdout, stderr) =
            (String::from_utf8_lossy(&output.stdout).into_owned(), String::from_utf8_lossy(&output.stderr).into_owned());
        let got = lines(&stdout);
        assert_eq!(got.len(), expected.len(), "{}: {stdout}
{stderr}", device.name);
        for (i, (g, e)) in got.iter().zip(&expected).take(4).enumerate() {
            let (g, e): (f64, f64) = (g.parse().unwrap(), e.parse().unwrap());
            assert!((g - e).abs() <= 1e-5 * e.abs(), "{}: product {}: {g} on the GPU, {e} on the CPU", device.name, i + 1);
        }
        assert_eq!(got[4..], expected[4..], "{}: the large product: stderr: {stderr}", device.name);
        assert!(!stderr.contains("cleave Vulkan validation"), "{}: {stderr}", device.name);
    }
}

/// Causal attention's gradient keeps its three results on the device across
/// heads: bufferization's deallocation takes their addresses each iteration
/// only to compare them (`dealloc_helper`), which isn't an access
/// (`addressOnlyCompared`); reported as one, each head re-uploaded them,
/// 14 GiB a step in nanoLM. Each head's scores, a fresh allocation reached
/// through a cast, are discarded rather than uploaded, and so are the three
/// results, fresh heap allocations made memrefs in the function
/// (`isFreshDescriptor`). Up: the four inputs, 256 KiB each, and scalar
/// kernel arguments; one submit per result read. Its loop over heads, a
/// `scf.while` for the deallocation's bookkeeping, is folded into its
/// kernels (`cleave-fold-loops-into-grids`), the bookkeeping left in a host
/// loop: 10 dispatches.
#[test]
fn attention_gradients_stay_on_the_device_across_heads() {
    let src = "
        use nn;
        use io;
        fn main() -> i32 {
            rand_seed(1);
            let q: Tensor<f32, 256, 256> = Init::he();
            let k: Tensor<f32, 256, 256> = Init::he();
            let v: Tensor<f32, 256, 256> = Init::he();
            let u: Tensor<f32, 256, 256> = Init::he();
            let (a, b, c) = CausalAttention::causal_attention_backward(q, k, v, AttentionShape::<256, 64>(), u);
            println(a[3, 5] + b[7, 9] + c[1, 2]);
            0
        }
    ";
    for (index, device) in devices().iter().enumerate() {
        let (stdout, stderr) = stats_on(index, "attention_backward", src);
        assert!(stdout.contains("main returned: 0"), "{}: stdout: {stdout}
stderr: {stderr}", device.name);
        assert!(!stderr.contains("cleave Vulkan validation"), "{}: {stderr}", device.name);
        assert!(
            stderr.contains(": 10 dispatches, 1048672 bytes uploaded, 786432 bytes downloaded, 3 submits"),
            "{}: {stderr}",
            device.name
        );
    }
}

/// A loop over rows whose body reduces each row runs a workgroup per row,
/// the row's elements shared among its threads and their partial maximum
/// and sum combined by a tree (`cleave-cooperative-rows`): 256 workgroups,
/// where a thread per row made 4. The sum is taken in another order than the
/// CPU's: equal within rounding.
#[test]
fn row_reductions_run_a_workgroup_per_row() {
    let src = "
        use nn;
        use io;
        #[no_inline]
        fn row_stats(x: Tensor<f32, 256, 1000>) -> Tensor<f32, 256, 2> {
            let mut y: [[f32; 2]; 256] = mlir::memref::alloc();
            for i in 0..256 {
                let mut m = x[i, 0];
                for j in 1..1000 { if x[i, j] > m { m = x[i, j]; }; };
                let mut s: f32 = 0.0;
                for j in 0..1000 { s = s + exp(x[i, j] - m); };
                y[i, 0] = m;
                y[i, 1] = s;
            };
            Tensor(data: y)
        }
        fn main() -> i32 {
            rand_seed(5);
            let x: Tensor<f32, 256, 1000> = Init::he();
            let y = row_stats(x);
            let mut t: f32 = 0.0;
            for i in 0..256 { t = t + y[i, 0] * 3.0 + y[i, 1]; };
            println(t);
            0
        }
    ";
    let devices = devices();
    if devices.is_empty() {
        return;
    }
    let dir = std::env::temp_dir().join(format!("cleave-vulkan-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join("row_stats.cleave");
    std::fs::write(&source, src).unwrap();
    let number = |out: &str| -> f64 { out.lines().next().and_then(|l| l.trim().parse().ok()).unwrap_or(f64::NAN) };
    let cpu = Command::new(env!("CARGO_BIN_EXE_cleave"))
        .args(["--no-openmp", "--no-debug-info", "--run"])
        .arg(&source)
        .output()
        .expect("cannot run cleave");
    let expected = number(&String::from_utf8_lossy(&cpu.stdout));
    assert!(expected.is_finite());
    for (index, _) in devices.iter().enumerate() {
        let output = Command::new(env!("CARGO_BIN_EXE_cleave"))
            .args(["--backend", "vulkan", "--no-openmp", "--no-debug-info", "--run"])
            .arg(&source)
            .env("CLEAVE_VK_DEVICE", index.to_string())
            .env("CLEAVE_VK_VALIDATION", "1")
            .env("CLEAVE_VK_PROFILE", "1")
            .output()
            .expect("cannot run cleave");
        let (stdout, stderr) =
            (String::from_utf8_lossy(&output.stdout).into_owned(), String::from_utf8_lossy(&output.stderr).into_owned());
        let got = number(&stdout);
        assert!((got - expected).abs() <= 1e-5 * expected.abs(), "{got} on the GPU, {expected} on the CPU
{stderr}");
        assert!(!stderr.contains("cleave Vulkan validation"), "{stderr}");
        assert!(stderr.contains("row_stats_kernel #0 [256, 1, 1]"), "{stderr}");
    }
}

/// A gradient summed over rows into one value per column (`rms_norm_grad_g`:
/// each row's scale computed, then `d[c] += u[r, c] * x[r, c] * scale`) runs
/// as three kernels (`interchangeAccumulation`): the zeros, each row's scale,
/// then each column summing its rows, a workgroup per column
/// (`promoteAccumulation`: a thread per column made 12 workgroups of nanoLM's
/// 768 columns, its costliest kernel). It ran a kernel per row with the
/// scales on the host, 1.5 GiB read back per nanoLM step. The sum is the
/// CPU's up to the device's square root and the rows' summing order.
#[test]
fn a_sum_over_rows_into_columns_runs_by_column() {
    let src = "
        use nn;
        use io;
        fn main() -> i32 {
            rand_seed(1);
            let x: Tensor<f32, 256, 256> = Init::he();
            let g: Tensor<f32, 1, 256> = Init::he();
            let u: Tensor<f32, 256, 256> = Init::he();
            let d = RmsNorm::rms_norm_grad_g(x, g, u);
            let mut s: f32 = 0.0;
            for c in 0..256 { s = s + d[0, c] * (c + 1).to(); };
            println(s);
            0
        }
    ";
    let devices = devices();
    if devices.is_empty() {
        return;
    }
    let dir = std::env::temp_dir().join(format!("cleave-vulkan-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join("column_sums.cleave");
    std::fs::write(&source, src).unwrap();
    let number = |out: &str| -> f64 { out.lines().next().and_then(|l| l.trim().parse().ok()).unwrap_or(f64::NAN) };
    let cpu = Command::new(env!("CARGO_BIN_EXE_cleave"))
        .args(["--no-openmp", "--no-debug-info", "--run"])
        .arg(&source)
        .output()
        .expect("cannot run cleave");
    let expected = number(&String::from_utf8_lossy(&cpu.stdout));
    assert!(expected.is_finite());
    for (index, device) in devices.iter().enumerate() {
        let output = Command::new(env!("CARGO_BIN_EXE_cleave"))
            .args(["--backend", "vulkan", "--no-openmp", "--no-debug-info", "--run"])
            .arg(&source)
            .env("CLEAVE_VK_DEVICE", index.to_string())
            .env("CLEAVE_VK_VALIDATION", "1")
            .env("CLEAVE_VK_PROFILE", "1")
            .output()
            .expect("cannot run cleave");
        let (stdout, stderr) =
            (String::from_utf8_lossy(&output.stdout).into_owned(), String::from_utf8_lossy(&output.stderr).into_owned());
        assert!(stderr.contains("Tensor<f32, 1, 256>>_kernel #2 [256, 1, 1]"), "{}: {stderr}", device.name);
        let got = number(&stdout);
        assert!((got - expected).abs() <= 1e-5 * expected.abs(), "{}: {got} on the GPU, {expected} on the CPU", device.name);
        assert!(!stderr.contains("cleave Vulkan validation"), "{}: {stderr}", device.name);
        assert!(stderr.contains(" 3 dispatches"), "{}: {stderr}", device.name);
    }
}

/// A bias's gradient (`reduce0`, a `linalg` sum over rows into a gradient's
/// field) runs a workgroup per column, as `rms_norm_grad_g` does: its
/// memory, cast again where the gradient is read or released, is fresh for
/// the loop that never reaches those other names (`freshInLoop`). A thread
/// per column made 8 workgroups here, 1.1 ms per nanoLM layer.
#[test]
fn a_bias_gradient_sums_its_rows_a_workgroup_per_column() {
    let src = "
        use io;
        use linalg;
        use nn;
        fn loss(layer: Dense<f32, 256, 512>, x: Tensor<f32, 1024, 256>) -> f32 {
            sum(layer.dense_forward(x))
        }
        dloss = grad(loss, layer);
        fn main() -> i32 {
            let x: Tensor<f32, 1024, 256> = [for i in 0..1024: [for j in 0..256: 0.25]];
            let layer = Dense::<f32, 256, 512>(w: [for i in 0..256: [for j in 0..512: 0.5]], b: [for i in 0..1: [for j in 0..512: 0.0]]);
            let g = dloss(layer, x);
            println(g.w[1, 2]);
            println(g.b[0, 3]);
            0
        }
    ";
    let numbers = |out: &str| -> Vec<f64> { out.lines().filter_map(|l| l.trim().parse().ok()).collect() };
    let expected = numbers(&run_cpu("bias_gradient", src));
    assert_eq!(expected, [256.0, 1024.0]);
    for (index, device) in devices().iter().enumerate() {
        let dir = std::env::temp_dir().join(format!("cleave-vulkan-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let source = dir.join("bias_gradient.cleave");
        std::fs::write(&source, src).unwrap();
        let output = Command::new(env!("CARGO_BIN_EXE_cleave"))
            .args(["--backend", "vulkan", "--no-openmp", "--no-debug-info", "--run"])
            .arg(&source)
            .env("CLEAVE_VK_DEVICE", index.to_string())
            .env("CLEAVE_VK_VALIDATION", "1")
            .env("CLEAVE_VK_PROFILE", "1")
            .output()
            .expect("cannot run cleave");
        let (stdout, stderr) =
            (String::from_utf8_lossy(&output.stdout).into_owned(), String::from_utf8_lossy(&output.stderr).into_owned());
        assert_eq!(numbers(&stdout), expected, "{}: {stdout}{stderr}", device.name);
        // The reduction's kernel and the one the forward broadcast makes.
        let per_column = stderr.lines().filter(|l| l.contains(" us ") && l.ends_with(" [512, 1, 1]")).count();
        assert_eq!(per_column, 2, "{}: no workgroup per column in {stderr}", device.name);
        assert!(!stderr.contains("cleave Vulkan validation"), "{}: {stderr}", device.name);
    }
}

/// A nest of loops whose iterations are independent at every level runs as
/// one kernel over all of them, a thread per innermost iteration, even where
/// an index combines several of them (`rope`'s `h * DH + j`, an affine
/// dependence test): it ran a thread per row, each turning 192 pairs, 16
/// workgroups for nanoLM's 1024 rows. A loop whose iterations read what the
/// one before wrote (a running sum along a row) stays sequential in its
/// row's thread.
#[test]
fn a_nest_of_independent_loops_runs_a_thread_per_innermost_iteration() {
    let src = "
        use nn;
        use io;
        #[no_inline]
        fn running_sums(x: Tensor<f32, 256, 128>) -> Tensor<f32, 256, 128> {
            let mut y: [[f32; 128]; 256] = mlir::memref::alloc();
            for r in 0..256 { y[r, 0] = x[r, 0]; };
            for r in 0..256 { for c in 1..128 { y[r, c] = y[r, c - 1] + x[r, c]; }; };
            Tensor(data: y)
        }
        fn main() -> i32 {
            rand_seed(3);
            let x: Tensor<f32, 256, 128> = Init::he();
            let y = rope(x, AttentionShape::<64, 32>());
            let z = running_sums(x);
            let mut s: f32 = 0.0;
            for r in 0..256 { for c in 0..128 { s = s + (y[r, c] + z[r, c] * 0.01) * (r * 3 + c).to(); }; };
            println(s);
            0
        }
    ";
    let devices = devices();
    if devices.is_empty() {
        return;
    }
    let dir = std::env::temp_dir().join(format!("cleave-vulkan-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join("rope.cleave");
    std::fs::write(&source, src).unwrap();
    let number = |out: &str| -> f64 { out.lines().next().and_then(|l| l.trim().parse().ok()).unwrap_or(f64::NAN) };
    let cpu = Command::new(env!("CARGO_BIN_EXE_cleave"))
        .args(["--no-openmp", "--no-debug-info", "--run"])
        .arg(&source)
        .output()
        .expect("cannot run cleave");
    let expected = number(&String::from_utf8_lossy(&cpu.stdout));
    assert!(expected.is_finite());
    for (index, device) in devices.iter().enumerate() {
        let output = Command::new(env!("CARGO_BIN_EXE_cleave"))
            .args(["--backend", "vulkan", "--no-openmp", "--no-debug-info", "--run"])
            .arg(&source)
            .env("CLEAVE_VK_DEVICE", index.to_string())
            .env("CLEAVE_VK_VALIDATION", "1")
            .env("CLEAVE_VK_PROFILE", "1")
            .output()
            .expect("cannot run cleave");
        let (stdout, stderr) =
            (String::from_utf8_lossy(&output.stdout).into_owned(), String::from_utf8_lossy(&output.stderr).into_owned());
        let got = number(&stdout);
        assert!((got - expected).abs() <= 1e-5 * expected.abs(), "{}: {got} on the GPU, {expected} on the CPU", device.name);
        assert!(!stderr.contains("cleave Vulkan validation"), "{}: {stderr}", device.name);
        assert!(stderr.contains("AttentionShape<64, 32>>_kernel #0 [256, 1, 1]"), "{}: {stderr}", device.name);
        assert!(stderr.contains("running_sums_kernel #2 [4, 1, 1]"), "{}: {stderr}", device.name);
    }
}

/// Causal attention skips the work the mask makes useless, found at compile
/// time (`cleave-tile-matmuls-for-gpu`): the tiles of `Q K^T` above the
/// diagonal, which the softmax never reads (120 of 256), forward and backward,
/// the same tiles of the backward's `dP = U V^T`, which the score gradient
/// reads only up to the diagonal (`attention_score_grad`), and in the products
/// by the weights or the score gradient (`P V`, `dQ`, `dK`, `dV`), the
/// slices of the reduction where those are zero (stored as 0, or as a
/// product with the zero weights). The same values as the CPU, forward and
/// backward.
#[test]
fn causal_attention_skips_the_masked_half() {
    let src = "
        use nn;
        use io;
        fn main() -> i32 {
            rand_seed(3);
            let q: Tensor<f32, 1024, 128> = Init::he();
            let k: Tensor<f32, 1024, 128> = Init::he();
            let v: Tensor<f32, 1024, 128> = Init::he();
            let u: Tensor<f32, 1024, 128> = Init::he();
            let o = causal_attention(q, k, v, AttentionShape::<1024, 64>());
            let (a, b, c) = CausalAttention::causal_attention_backward(q, k, v, AttentionShape::<1024, 64>(), u);
            for r in [0, 5, 63, 64, 517, 1000, 1023] {
                for col in [0, 33, 64, 127] {
                    println(o[r, col]);
                    println(a[r, col]);
                    println(b[r, col]);
                    println(c[r, col]);
                };
            };
            0
        }
    ";
    let devices = devices();
    if devices.is_empty() {
        return;
    }
    let dir = std::env::temp_dir().join(format!("cleave-vulkan-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join("masked_attention.cleave");
    std::fs::write(&source, src).unwrap();
    let numbers = |out: &str| -> Vec<f64> { out.lines().filter_map(|l| l.trim().parse().ok()).collect() };
    let cpu = Command::new(env!("CARGO_BIN_EXE_cleave"))
        .args(["--no-openmp", "--no-debug-info", "--run"])
        .arg(&source)
        .output()
        .expect("cannot run cleave");
    let expected = numbers(&String::from_utf8_lossy(&cpu.stdout));
    assert_eq!(expected.len(), 7 * 4 * 4);
    let scale = expected.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    for (index, device) in devices.iter().enumerate() {
        let output = Command::new(env!("CARGO_BIN_EXE_cleave"))
            .args(["--backend", "vulkan", "--no-openmp", "--no-debug-info", "--run"])
            .arg(&source)
            .env("CLEAVE_VK_DEVICE", index.to_string())
            .env("CLEAVE_VK_VALIDATION", "1")
            .env("CLEAVE_DEBUG_PARALLELIZE", "1")
            .output()
            .expect("cannot run cleave");
        let (stdout, stderr) =
            (String::from_utf8_lossy(&output.stdout).into_owned(), String::from_utf8_lossy(&output.stderr).into_owned());
        let got = numbers(&stdout);
        assert_eq!(got.len(), expected.len(), "{}: {stdout}", device.name);
        for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
            assert!((g - e).abs() <= 1e-4 * scale, "{}: value {i}: {g} on the GPU, {e} on the CPU", device.name);
        }
        assert!(!stderr.contains("cleave Vulkan validation"), "{}: {stderr}", device.name);
        assert_eq!(stderr.matches(", 120 dead tiles of 256").count(), 3, "{}: {stderr}", device.name);
        assert_eq!(stderr.matches("some zero slices").count(), 4, "{}: {stderr}", device.name);
    }
}

/// A host loop whose iterations' allocations don't all fit the device's
/// cache together is folded into its kernels by groups
/// (`cleave-fold-loops-into-grids`, `CLEAVE_GPU_FOLD_BYTES`), a host loop over
/// the groups left: 4 heads holding 8 MiB each at most, by 2 under 24 MiB. The CPU's
/// values.
#[test]
fn a_host_loop_folds_into_its_kernels_by_groups() {
    let src = "
        use nn;
        use io;
        fn main() -> i32 {
            rand_seed(4);
            let q: Tensor<f32, 1024, 256> = Init::he();
            let k: Tensor<f32, 1024, 256> = Init::he();
            let v: Tensor<f32, 1024, 256> = Init::he();
            let o = causal_attention(q, k, v, AttentionShape::<1024, 64>());
            for r in [0, 9, 511, 1023] {
                for c in [0, 70, 130, 255] { println(o[r, c]); };
            };
            0
        }
    ";
    let devices = devices();
    if devices.is_empty() {
        return;
    }
    let dir = std::env::temp_dir().join(format!("cleave-vulkan-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join("grouped_heads.cleave");
    std::fs::write(&source, src).unwrap();
    let numbers = |out: &str| -> Vec<f64> { out.lines().filter_map(|l| l.trim().parse().ok()).collect() };
    let cpu = Command::new(env!("CARGO_BIN_EXE_cleave"))
        .args(["--no-openmp", "--no-debug-info", "--run"])
        .arg(&source)
        .output()
        .expect("cannot run cleave");
    let expected = numbers(&String::from_utf8_lossy(&cpu.stdout));
    assert_eq!(expected.len(), 16);
    let scale = expected.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    for (index, device) in devices.iter().enumerate() {
        let output = Command::new(env!("CARGO_BIN_EXE_cleave"))
            .args(["--backend", "vulkan", "--no-openmp", "--no-debug-info", "--run"])
            .arg(&source)
            .env("CLEAVE_VK_DEVICE", index.to_string())
            .env("CLEAVE_VK_VALIDATION", "1")
            .env("CLEAVE_DEBUG_PARALLELIZE", "1")
            .env("CLEAVE_GPU_FOLD_BYTES", (24 << 20).to_string())
            .output()
            .expect("cannot run cleave");
        let (stdout, stderr) =
            (String::from_utf8_lossy(&output.stdout).into_owned(), String::from_utf8_lossy(&output.stderr).into_owned());
        let got = numbers(&stdout);
        assert_eq!(got.len(), expected.len(), "{}: {stdout}", device.name);
        for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
            assert!((g - e).abs() <= 1e-4 * scale, "{}: value {i}: {g} on the GPU, {e} on the CPU", device.name);
        }
        assert!(!stderr.contains("cleave Vulkan validation"), "{}: {stderr}", device.name);
        assert!(stderr.contains("folded 4 iterations by 2"), "{}: {stderr}", device.name);
    }
}

/// An embedding's gradient, each upstream row added into the table row its
/// id picks (rows hit by several ids), runs by column, each column's rows in
/// their order (`interchangeAccumulation`): the CPU's values to the bit, in
/// 4 dispatches. It ran a kernel per row, the id read on the host before
/// each (8192 dispatches and host synchronizations a nanoLM step).
#[test]
fn an_embedding_gradient_runs_by_column() {
    let src = "
        use nn;
        use io;
        fn main() -> i32 {
            rand_seed(5);
            let table: Tensor<f32, 512, 128> = Init::he();
            let u: Tensor<f32, 1024, 128> = Init::he();
            let ids = [for n in 0..1024: (n * 37 + n / 3) - ((n * 37 + n / 3) / 512) * 512];
            let g = Embed::embed_grad(table, ids, u);
            for r in [0, 5, 77, 300, 511] {
                for c in [0, 64, 127] { println(g[r, c]); };
            };
            0
        }
    ";
    let devices = devices();
    if devices.is_empty() {
        return;
    }
    let dir = std::env::temp_dir().join(format!("cleave-vulkan-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join("embedding_gradient.cleave");
    std::fs::write(&source, src).unwrap();
    let cpu = Command::new(env!("CARGO_BIN_EXE_cleave"))
        .args(["--no-openmp", "--no-debug-info", "--run"])
        .arg(&source)
        .output()
        .expect("cannot run cleave");
    let expected = String::from_utf8_lossy(&cpu.stdout).into_owned();
    assert!(expected.contains("main returned: 0"), "cpu: {expected}");
    for (index, device) in devices.iter().enumerate() {
        let (stdout, stderr) = stats_on(index, "embedding_gradient", src);
        assert_eq!(stdout, expected, "{}: {stderr}", device.name);
        assert!(!stderr.contains("cleave Vulkan validation"), "{}: {stderr}", device.name);
        assert!(stderr.contains(": 4 dispatches"), "{}: {stderr}", device.name);
    }
}

/// A matrix's sum is its rows' sums, a kernel (a workgroup per row), added
/// up by the host: 512 floats read back rather than the matrix (nanoLM's
/// gradient clipping read its squared gradients back, 433 MiB a step). The
/// float sum is reassociated, closer to the exact one than the CPU's
/// sequential one: 1e-6 of a sum in f64, which the CPU's misses by 1.6e-4.
#[test]
fn a_matrix_sum_reads_back_its_rows_sums() {
    let src = "
        use nn;
        use io;
        fn main() -> i32 {
            rand_seed(2);
            let x: Tensor<f32, 512, 1000> = Init::he();
            println(sum(x * x));
            0
        }
    ";
    let exact_src = "
        use nn;
        use io;
        fn main() -> i32 {
            rand_seed(2);
            let x: Tensor<f32, 512, 1000> = Init::he();
            let mut t: f64 = 0.0;
            for i in 0..512 { for j in 0..1000 { let v: f64 = (x[i, j] * x[i, j]).to(); t = t + v; }; };
            println(t);
            0
        }
    ";
    let devices = devices();
    if devices.is_empty() {
        return;
    }
    let dir = std::env::temp_dir().join(format!("cleave-vulkan-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join("exact_sum.cleave");
    std::fs::write(&source, exact_src).unwrap();
    let number = |out: &str| -> f64 { out.lines().next().and_then(|l| l.trim().parse().ok()).unwrap_or(f64::NAN) };
    let cpu = Command::new(env!("CARGO_BIN_EXE_cleave"))
        .args(["--no-openmp", "--no-debug-info", "--run"])
        .arg(&source)
        .output()
        .expect("cannot run cleave");
    let exact = number(&String::from_utf8_lossy(&cpu.stdout));
    assert!(exact.is_finite());
    for (index, device) in devices.iter().enumerate() {
        let (stdout, stderr) = stats_on(index, "matrix_sum", src);
        let got = number(&stdout);
        assert!((got - exact).abs() <= 1e-6 * exact, "{}: {got} on the GPU, {exact} exactly", device.name);
        assert!(!stderr.contains("cleave Vulkan validation"), "{}: {stderr}", device.name);
        assert!(stderr.contains(": 2 dispatches") && stderr.contains(" 2048 bytes downloaded"), "{}: {stderr}", device.name);
    }
}

/// Every example prints on the GPU what it prints on the CPU, on every
/// device, the validation layer silent: what found, in Phase 1, a kernel
/// bound to a view's first element while it indexes from the buffer's start,
/// an upload skipped for a kernel writing only part of what it's bound to,
/// an address handed to an extern (`sgemm`) unseen by the host
/// synchronization, `math.exp` and scalar kernel arguments unsupported.
/// `xor` is left out: its thousands of tiny kernels take a minute until
/// placement keeps small work on the CPU (Phase 5).
#[test]
fn examples_print_the_same_on_the_gpu() {
    let devices = devices();
    if devices.is_empty() {
        return;
    }
    let examples = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../examples");
    let mut sources: Vec<std::path::PathBuf> = std::fs::read_dir(&examples)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "cleave") && p.file_stem().is_some_and(|s| s != "xor"))
        .collect();
    sources.sort();
    assert!(sources.len() >= 10, "examples not found in {}", examples.display());
    for source in &sources {
        let run = |backend: &str, device: usize| {
            let output = Command::new(env!("CARGO_BIN_EXE_cleave"))
                .args(["--backend", backend, "--no-openmp", "--no-debug-info", "--run"])
                .arg(source)
                .env("CLEAVE_VK_DEVICE", device.to_string())
                .env("CLEAVE_VK_VALIDATION", "1")
                .output()
                .expect("cannot run cleave");
            (String::from_utf8_lossy(&output.stdout).into_owned(), String::from_utf8_lossy(&output.stderr).into_owned())
        };
        let (cpu, _) = run("cpu", 0);
        for (index, device) in devices.iter().enumerate() {
            let (gpu, stderr) = run("vulkan", index);
            assert_eq!(gpu, cpu, "{} on {}: stderr: {stderr}", source.display(), device.name);
            assert!(!stderr.contains("cleave Vulkan validation"), "{} on {}: {stderr}", source.display(), device.name);
        }
    }
}

/// Tensors held in a struct's fields stay on the device too: a layer's `w`
/// and `b`, read out of the struct at each of five calls, are uploaded once,
/// not at each call: the module's descriptors all view heap payloads
/// (`descriptorsAreHeapPayloads`).
#[test]
fn struct_field_tensors_stay_on_the_device() {
    let src = "
        use nn;

        struct Layer {
            w: Tensor<f32, 4, 256>,
            b: Tensor<f32, 4, 256>,
        }

        #[no_inline]
        fn apply(l: Layer, x: Tensor<f32, 4, 256>) -> Tensor<f32, 4, 256> {
            x * l.w + l.b
        }

        fn main() -> i32 {
            rand_seed(1);
            let l = Layer(w: Init::he(), b: Init::he());
            let mut x: Tensor<f32, 4, 256> = Init::he();
            let x0 = x[1, 7];
            for i in 0..5 {
                x = apply(l, x);
            };
            let mut e = x0;
            for i in 0..5 {
                e = e * l.w[1, 7] + l.b[1, 7];
            };
            let d = x[1, 7] - e;
            if d < 0.001 and d > -0.001 { 0 } else { 1 }
        }
    ";
    for (index, device) in devices().iter().enumerate() {
        let (stdout, stderr) = stats_on(index, "struct_fields", src);
        assert!(stdout.contains("main returned: 0"), "{}: stdout: {stdout}
stderr: {stderr}", device.name);
        assert!(!stderr.contains("cleave Vulkan validation"), "{}: {stderr}", device.name);
        // `x`, `w`, `b` once (12 KiB), and the struct's two fields' storage once
        // when it's built (8 KiB): copied into by a kernel, but a descriptor
        // built from `cleave_alloc_rc`, not a `memref.alloc`, so not known
        // fresh (`isFreshUnaliased`). Not once per call.
        assert!(stderr.contains("20480 bytes uploaded"), "{}: {stderr}", device.name);
    }
}

/// A kernel writing a view at an offset known only at run time (`update`'s
/// row `r`): the view and its index arithmetic moved into the kernel, folded
/// into its accesses, so that the kernel takes the whole tensor (SPIR-V can't
/// express a dynamic offset in a buffer's type); the rows it doesn't write
/// keep the host's values, since only an op overwriting a whole payload
/// skips its upload (`cleave_vk_discard`).
#[test]
fn a_kernel_writing_a_row_at_a_run_time_offset() {
    let src = "
        use nn;

        #[no_inline]
        fn put(x: Tensor<f32, 8, 64>, row: Tensor<f32, 1, 64>, r: i32) -> Tensor<f32, 8, 64> {
            update(x, row, r, 0)
        }

        fn main() -> i32 {
            rand_seed(1);
            let x: Tensor<f32, 8, 64> = Init::he();
            let row: Tensor<f32, 1, 64> = Init::he();
            let y = put(x, row, 3);
            let d = y[3, 10] - row[0, 10];
            if d == 0.0 and y[2, 5] == x[2, 5] { 0 } else { 1 }
        }
    ";
    for (index, device) in devices().iter().enumerate() {
        let (stdout, stderr) = run_on(index, "update_row", src);
        assert!(stdout.contains("main returned: 0"), "{}: stdout: {stdout}
stderr: {stderr}", device.name);
        assert!(!stderr.contains("cleave Vulkan validation"), "{}: {stderr}", device.name);
    }
}

/// A miniature nanoLM (a block of attention and SwiGLU, `grad` of the
/// loss): the loss and every parameter's gradient (sums of squares) agree
/// between the CPU and each device, within a relative 1e-4. Found, in
/// Phase 3, an in-place elementwise op (`o * w` writing into `o`) declared
/// a full overwrite of a buffer one of its inputs reads: the host's values
/// were dropped (`cleave_vk_discard` now only for fresh allocations no input
/// reads). nanoLM's attention gradients were 3 to 15 times off. Again with
/// 40 KB of device memory for mirrors (`CLEAVE_VK_DEVICE_BUDGET`): the
/// least recently used evicted, brought home first, uploaded again when
/// bound, the same gradients (nanoLM at its full size outgrows the 4070
/// SUPER's 12 GiB).
#[test]
fn a_miniature_transformer_has_the_cpus_gradients() {
    let src = "
        use num;
        use convert;
        use nn;
        use io;

        define WIDTH: i32 = 64;
        define LAYERS: i32 = 1;
        define HEAD: i32 = 32;
        define HIDDEN: i32 = 128;
        define CONTEXT: i32 = 16;
        define ROWS: i32 = 16;
        define VOCAB: i32 = 32;

        struct Block {
            n1: Tensor<f32, 1, WIDTH>,
            wq: Dense<f32, WIDTH, WIDTH>, wk: Dense<f32, WIDTH, WIDTH>, wv: Dense<f32, WIDTH, WIDTH>, wo: Dense<f32, WIDTH, WIDTH>,
            n2: Tensor<f32, 1, WIDTH>,
            mlp: SwiGlu<f32, WIDTH, HIDDEN>
        }
        impl Trainable<Block> {}

        struct Gpt {
            tok: Embedding<f32, VOCAB, WIDTH>,
            blocks: [Block; LAYERS],
            nf: Tensor<f32, 1, WIDTH>
        }
        impl Trainable<Gpt> {}

        fn ones() -> Tensor<f32, 1, WIDTH> { [for i in 0..1: [for j in 0..WIDTH: 1.0]] }

        fn new_block() -> Block {
            Block(n1: ones(), wq: Init::xavier(), wk: Init::xavier(), wv: Init::xavier(), wo: Init::xavier(), n2: ones(), mlp: Init::xavier())
        }

        #[no_inline]
        fn block<const N: i32>(x: Tensor<f32, N, WIDTH>, b: Block) -> Tensor<f32, N, WIDTH> {
            let shape = AttentionShape::<CONTEXT, HEAD>();
            let h = rms_norm(x, b.n1);
            let q = rope(b.wq.dense_forward(h), shape);
            let k = rope(b.wk.dense_forward(h), shape);
            let a = causal_attention(q, k, b.wv.dense_forward(h), shape);
            let x2 = x + b.wo.dense_forward(a);
            x2 + b.mlp.swiglu_forward(rms_norm(x2, b.n2))
        }

        fn gpt_logits<const N: i32>(x: [i32; N], m: Gpt) -> Tensor<f32, N, VOCAB> {
            let mut h = embedding_forward(m.tok, x);
            for i in 0..LAYERS { h = block(h, m.blocks[i]); };
            embedding_logits(m.tok, rms_norm(h, m.nf))
        }

        fn loss(x: [i32; ROWS], y: [i32; ROWS], m: Gpt) -> f32 {
            sparse_cross_entropy(gpt_logits(x, m), y)
        }

        g = grad(loss, m);

        fn sq<const R: i32, const C: i32>(t: Tensor<f32, R, C>) -> f32 { sum(t * t) }

        fn main() -> i32 {
            rand_seed(7);
            let m = Gpt(tok: Init::xavier(), blocks: [for i in 0..LAYERS: new_block()], nf: ones());
            let x: [i32; ROWS] = [for i in 0..ROWS: mod(i * 7 + 3, VOCAB)];
            let y: [i32; ROWS] = [for i in 0..ROWS: mod(i * 5 + 1, VOCAB)];
            println(loss(x, y, m));
            let d = g(x, y, m);
            println(sq(d.tok.table));
            println(sq(d.nf));
            println(sq(d.blocks[0].n1));
            println(sq(d.blocks[0].wq.w));
            println(sq(d.blocks[0].wk.w));
            println(sq(d.blocks[0].wv.w));
            println(sq(d.blocks[0].wo.w));
            println(sq(d.blocks[0].n2));
            println(sq(d.blocks[0].mlp.gate.w));
            println(sq(d.blocks[0].mlp.up.w));
            println(sq(d.blocks[0].mlp.down.w));
            0
        }
    ";
    let devices = devices();
    if devices.is_empty() {
        return;
    }
    let dir = std::env::temp_dir().join(format!("cleave-vulkan-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join("tiny_gpt.cleave");
    std::fs::write(&source, src).unwrap();
    let numbers = |stdout: &str| -> Vec<f64> { stdout.lines().filter_map(|l| l.trim().parse().ok()).collect() };
    // `budget`: a device memory budget (`CLEAVE_VK_DEVICE_BUDGET`) small
    // enough that mirrors are evicted and uploaded again.
    let run = |backend: &str, device: usize, budget: Option<&str>| {
        let mut command = Command::new(env!("CARGO_BIN_EXE_cleave"));
        command
            .args(["--backend", backend, "--no-openmp", "--no-debug-info", "--run"])
            .arg(&source)
            .env("CLEAVE_VK_DEVICE", device.to_string())
            .env("CLEAVE_VK_VALIDATION", "1");
        if let Some(budget) = budget {
            command.env("CLEAVE_VK_DEVICE_BUDGET", budget).env("CLEAVE_VK_PROFILE", "1");
        }
        let output = command.output().expect("cannot run cleave");
        (String::from_utf8_lossy(&output.stdout).into_owned(), String::from_utf8_lossy(&output.stderr).into_owned())
    };
    let (cpu, _) = run("cpu", 0, None);
    let expected = numbers(&cpu);
    assert_eq!(expected.len(), 12, "cpu: {cpu}");
    for ((index, device), budget) in
        devices.iter().enumerate().flat_map(|d| [(d, None), (d, Some("40000"))])
    {
        let (gpu, stderr) = run("vulkan", index, budget);
        if budget.is_some() {
            let evicted = stderr
                .split(" mirrors evicted")
                .next()
                .and_then(|before| before.rsplit(' ').next())
                .and_then(|n| n.parse::<u64>().ok());
            assert!(evicted.is_some_and(|n| n > 0), "{}: no mirror evicted under the budget: {stderr}", device.name);
        }
        let got = numbers(&gpu);
        assert_eq!(got.len(), expected.len(), "{}: {gpu}
{stderr}", device.name);
        for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
            assert!((g - e).abs() <= 1e-4 * e.abs().max(1.0), "{}: value {i}: {g} on the GPU, {e} on the CPU", device.name);
        }
        assert!(!stderr.contains("cleave Vulkan validation"), "{}: {stderr}", device.name);
    }
}

/// A `spawn` on the Vulkan backend runs in place, as an ordinary call (no host
/// tasks there), and says so under `CLEAVE_DEBUG_PARALLELIZE=1`, once per site
/// with its count (`doc/plan-spawn-gpu.md`, step 1): the mapping is reported,
/// never silent. A comprehension's spawns, summed over the tuple by an
/// unrolled loop, give the CPU's value.
#[test]
fn a_spawn_on_the_gpu_reports_that_it_runs_in_place() {
    let src = "
        use io;
        struct P { a: f32, b: f32 }
        fn make(x: f32) -> P { P(a: x, b: 2.0 * x) }
        fn add(p: P, q: P) -> P { P(a: p.a + q.a, b: p.b + q.b) }
        fn main() -> i32 {
            let g = [for j in 0..4: spawn make(1.5)];
            let mut acc = g[0];
            for j in 1..4 { acc = add(acc, g[j]); };
            println(acc.b);
            0
        }
    ";
    let devices = devices();
    if devices.is_empty() {
        return;
    }
    let dir = std::env::temp_dir().join(format!("cleave-vulkan-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join("spawn_in_place.cleave");
    std::fs::write(&source, src).unwrap();
    for (index, device) in devices.iter().enumerate() {
        let output = Command::new(env!("CARGO_BIN_EXE_cleave"))
            .args(["--backend", "vulkan", "--no-openmp", "--no-debug-info", "--run"])
            .arg(&source)
            .env("CLEAVE_VK_DEVICE", index.to_string())
            .env("CLEAVE_DEBUG_PARALLELIZE", "1")
            .output()
            .expect("cannot run cleave");
        let (stdout, stderr) =
            (String::from_utf8_lossy(&output.stdout).into_owned(), String::from_utf8_lossy(&output.stderr).into_owned());
        assert!(stdout.lines().any(|l| l.trim() == "12"), "{}: {stdout}", device.name);
        assert_eq!(
            stderr.matches("cleave-spawn: main: `make` runs in place (4 spawns): no host tasks on the Vulkan backend").count(),
            1,
            "{}: {stderr}",
            device.name
        );
    }
}

/// Each device's cooperative-matrix products (`VK_KHR_cooperative_matrix`), printed: what a
/// tensor-core kernel can be built from (`doc/plan-quantization.md`, phase 0). Every listed product
/// has positive sizes and a known scope; a device with the extension lists at least one f16 product
/// (the extension's baseline on every driver seen so far).
#[test]
fn cooperative_matrix_shapes_are_listed_per_device() {
    for device in devices() {
        eprintln!("{}: {} cooperative-matrix products", device.name, device.cooperative_matrices.len());
        for shape in &device.cooperative_matrices {
            eprintln!("  {shape}");
            assert!(shape.m > 0 && shape.n > 0 && shape.k > 0, "{}: {shape}", device.name);
            assert!(!shape.scope.starts_with("scope#"), "{}: {shape}", device.name);
        }
        if !device.cooperative_matrices.is_empty() {
            assert!(device.cooperative_matrices.iter().any(|s| s.a == "f16"), "{}: no f16 product", device.name);
        }
    }
}

/// `bf16` tensors on the GPU round as the CPU does: SPIR-V has no bf16 arithmetic, so kernels compute
/// in f32 and round each result to bf16 (`arith-emulate-unsupported-floats`, the bf16 capabilities
/// declared; without them MLIR silently held bf16 values as f32, and `1 + 2^-9` stayed `1 + 2^-9`).
/// Products of bf16 tensors accumulate in f32 (259 ones: the bf16 260, not 256). Devices without bf16
/// in shaders are skipped, saying so.
#[test]
fn bf16_tensors_round_on_the_gpu_as_on_the_cpu() {
    let src = "
        use io;
        use linalg;
        use convert;
        fn main() -> i32 {
            let a: Tensor<f32, 64, 256> = [for i in 0..64: [for j in 0..256: 1.001953125]];
            let h: Tensor<bf16, 64, 256> = a.to();
            let back: Tensor<f32, 64, 256> = h.to();
            println(back[3, 10]);
            let twice: Tensor<f32, 64, 256> = (h + h).to();
            println(twice[3, 10]);
            let x: Tensor<bf16, 1, 259> = [for i in 0..1: [for j in 0..259: 1.0]];
            let y: Tensor<bf16, 259, 1> = [for i in 0..259: [for j in 0..1: 1.0]];
            let s: f32 = matmul(x, y)[0, 0].to();
            println(s);
            0
        }
    ";
    let numbers = |out: &str| -> Vec<f64> { out.lines().filter_map(|l| l.trim().parse().ok()).collect() };
    assert_eq!(numbers(&run_cpu("bf16_rounding", src)), vec![1.0, 2.0, 260.0]);
    for (index, device) in devices().iter().enumerate() {
        if !device.bf16 {
            eprintln!("skipped on {}: no bf16 in shaders", device.name);
            continue;
        }
        let (stdout, stderr) = run_on(index, "bf16_rounding", src);
        assert_eq!(numbers(&stdout), vec![1.0, 2.0, 260.0], "{}: {stdout}{stderr}", device.name);
        assert!(!stderr.contains("cleave Vulkan validation"), "{}: {stderr}", device.name);
    }
}

/// Products asked in bfloat16 (`matmul_bf16` and its transposed forms, `stdlib/linalg`) run on the
/// tensor cores (`cleave-lower-gpu-gemms`, `cleave_vk_gemm_bf16`: the profile names the kernel) and
/// give the CPU's values: operands rounded to bf16, f32 accumulation. `1 + 2^-9` rounds to 1 in
/// bf16, so the bf16 product (128) differs from the f32 one (128.25) on purpose. On devices without
/// bf16 cooperative matrices, skipped.
#[test]
fn bf16_products_run_on_tensor_cores() {
    let src = "
        use io;
        use linalg;
        use convert;
        fn main() -> i32 {
            let a: Tensor<f32, 256, 128> = [for i in 0..256: [for j in 0..128: 1.001953125]];
            let b: Tensor<f32, 128, 128> = [for i in 0..128: [for j in 0..128: rem(i + j, 3).to()]];
            println(matmul_bf16(a, b)[3, 5]);
            println(matmul(a, b)[3, 5]);
            println(matmul_bf16_transpose_a(a, a)[2, 9]);
            println(matmul_bf16_transpose_b(b, b)[4, 4]);
            0
        }
    ";
    let numbers = |out: &str| -> Vec<f64> { out.lines().filter_map(|l| l.trim().parse().ok()).collect() };
    assert_eq!(numbers(&run_cpu("bf16_products", src)), vec![128.0, 128.25, 256.0, 215.0]);
    for (index, device) in devices().iter().enumerate() {
        if !device.cooperative_matrices.iter().any(|s| s.a == "bf16" && s.c == "f32") {
            eprintln!("skipped on {}: no bf16 cooperative matrices", device.name);
            continue;
        }
        let dir = std::env::temp_dir().join(format!("cleave-vulkan-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let source = dir.join("bf16_products.cleave");
        std::fs::write(&source, src).unwrap();
        let output = Command::new(env!("CARGO_BIN_EXE_cleave"))
            .args(["--backend", "vulkan", "--no-openmp", "--no-debug-info", "--run"])
            .arg(&source)
            .env("CLEAVE_VK_DEVICE", index.to_string())
            .env("CLEAVE_VK_VALIDATION", "1")
            .env("CLEAVE_VK_PROFILE", "1")
            .output()
            .expect("cannot run cleave");
        let (stdout, stderr) =
            (String::from_utf8_lossy(&output.stdout).into_owned(), String::from_utf8_lossy(&output.stderr).into_owned());
        assert_eq!(numbers(&stdout), vec![128.0, 128.25, 256.0, 215.0], "{}: {stdout}{stderr}", device.name);
        for kernel in ["gemm_bf16<256x128x128>", "gemm_bf16<128x128x256, A^T>", "gemm_bf16<128x128x128, B^T>"] {
            assert!(stderr.contains(kernel), "{}: no {kernel} in {stderr}", device.name);
        }
        assert!(!stderr.contains("cleave Vulkan validation"), "{}: {stderr}", device.name);
    }
}

/// A product of bf16 tensors runs on the tensor cores with its operands read as bf16
/// (`cleave-lower-gpu-gemms`, `cleave_vk_gemm_bf16_operands`), every transposition, and gives the
/// CPU's values (f32 accumulation); one whose shape the kernel can't take stays a generated kernel,
/// same values. Devices without bf16 cooperative matrices skipped.
#[test]
fn bf16_tensor_products_run_on_tensor_cores() {
    let src = "
        use io;
        use linalg;
        use convert;
        fn main() -> i32 {
            let m: Tensor<bf16, 512, 256> = [for i in 0..512: [for j in 0..256: rem(i + j, 3).to()]];
            let n: Tensor<bf16, 256, 384> = [for i in 0..256: [for j in 0..384: rem(i * j, 2).to()]];
            let c: Tensor<f32, 512, 384> = matmul(m, n).to();
            println(c[5, 7]);
            let ta: Tensor<f32, 256, 256> = matmul_transpose_a(m, m).to();
            println(ta[3, 4]);
            let tb: Tensor<f32, 384, 384> = matmul_transpose_b(n.transpose(), n.transpose()).to();
            println(tb[1, 2]);
            let x: Tensor<bf16, 1, 259> = [for i in 0..1: [for j in 0..259: 1.0]];
            let y: Tensor<bf16, 259, 1> = [for i in 0..259: [for j in 0..1: 1.0]];
            let s: f32 = matmul(x, y)[0, 0].to();
            println(s);
            0
        }
    ";
    let numbers = |out: &str| -> Vec<f64> { out.lines().filter_map(|l| l.trim().parse().ok()).collect() };
    let expected = numbers(&run_cpu("bf16_tensor_products", src));
    assert_eq!(expected.len(), 4, "{expected:?}");
    assert_eq!(expected[3], 260.0);
    for (index, device) in devices().iter().enumerate() {
        if !device.cooperative_matrices.iter().any(|s| s.a == "bf16" && s.c == "f32") {
            eprintln!("skipped on {}: no bf16 cooperative matrices", device.name);
            continue;
        }
        let dir = std::env::temp_dir().join(format!("cleave-vulkan-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let source = dir.join("bf16_tensor_products.cleave");
        std::fs::write(&source, src).unwrap();
        let output = Command::new(env!("CARGO_BIN_EXE_cleave"))
            .args(["--backend", "vulkan", "--no-openmp", "--no-debug-info", "--run"])
            .arg(&source)
            .env("CLEAVE_VK_DEVICE", index.to_string())
            .env("CLEAVE_VK_VALIDATION", "1")
            .env("CLEAVE_VK_PROFILE", "1")
            .output()
            .expect("cannot run cleave");
        let (stdout, stderr) =
            (String::from_utf8_lossy(&output.stdout).into_owned(), String::from_utf8_lossy(&output.stderr).into_owned());
        assert_eq!(numbers(&stdout), expected, "{}: {stdout}{stderr}", device.name);
        for kernel in [
            "gemm_bf16_operands<512x384x256>",
            "gemm_bf16_operands<256x256x512, A^T>",
            "gemm_bf16_operands<384x384x256, B^T>",
        ] {
            assert!(stderr.contains(kernel), "{}: no {kernel} in {stderr}", device.name);
        }
        assert!(!stderr.contains("cleave Vulkan validation"), "{}: {stderr}", device.name);
    }
}

// `matmul_wide`'s gradient on the GPU: the forward product and both adjoint
// products (`matmul_wide_transpose_a/b`) on the bf16-operand GEMM, the values
// the CPU computes.
#[test]
fn wide_bf16_products_differentiate_on_tensor_cores() {
    let src = "
        use io;
        use linalg;
        use convert;
        use nn;
        fn loss(x: Tensor<f32, 256, 128>, w: Tensor<f32, 128, 384>) -> f32 {
            let xb: Tensor<bf16, 256, 128> = x.to();
            let wb: Tensor<bf16, 128, 384> = w.to();
            sum(matmul_wide(xb, wb))
        }
        dloss_w = grad(loss, w);
        dloss_x = grad(loss, x);
        fn main() -> i32 {
            let x: Tensor<f32, 256, 128> = [for i in 0..256: [for j in 0..128: 1.001953125]];
            let w: Tensor<f32, 128, 384> = [for i in 0..128: [for j in 0..384: 0.5]];
            println(loss(x, w));
            println(dloss_w(x, w)[17, 99]);
            println(dloss_x(x, w)[3, 4]);
            0
        }
    ";
    let numbers = |out: &str| -> Vec<f64> { out.lines().filter_map(|l| l.trim().parse().ok()).collect() };
    let expected = numbers(&run_cpu("wide_bf16_products", src));
    assert_eq!(expected, [6291456.0, 256.0, 192.0]);
    for (index, device) in devices().iter().enumerate() {
        if !device.cooperative_matrices.iter().any(|s| s.a == "bf16" && s.c == "f32") {
            eprintln!("skipped on {}: no bf16 cooperative matrices", device.name);
            continue;
        }
        let dir = std::env::temp_dir().join(format!("cleave-vulkan-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let source = dir.join("wide_bf16_products.cleave");
        std::fs::write(&source, src).unwrap();
        let output = Command::new(env!("CARGO_BIN_EXE_cleave"))
            .args(["--backend", "vulkan", "--no-openmp", "--no-debug-info", "--run"])
            .arg(&source)
            .env("CLEAVE_VK_DEVICE", index.to_string())
            .env("CLEAVE_VK_VALIDATION", "1")
            .env("CLEAVE_VK_PROFILE", "1")
            .output()
            .expect("cannot run cleave");
        let (stdout, stderr) =
            (String::from_utf8_lossy(&output.stdout).into_owned(), String::from_utf8_lossy(&output.stderr).into_owned());
        assert_eq!(numbers(&stdout), expected, "{}: {stdout}{stderr}", device.name);
        for kernel in [
            "gemm_bf16_operands<256x384x128>",
            "gemm_bf16_operands<128x384x256, A^T>",
            "gemm_bf16_operands<256x128x384, B^T>",
        ] {
            assert!(stderr.contains(kernel), "{}: no {kernel} in {stderr}", device.name);
        }
        assert!(!stderr.contains("cleave Vulkan validation"), "{}: {stderr}", device.name);
    }
}
