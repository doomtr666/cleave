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

/// Causal attention moves only what it must: its three inputs and its
/// out-parameter go up once (the out-parameter because the per-head blocks
/// are views of it), every head's scores and weights stay on the device, a
/// fresh allocation's undefined contents never uploaded (`causal_softmax`'s
/// weights, which its kernel reads back), and the host reads one element
/// back. Up: four 256 KiB tensors and 80 bytes of scalar kernel arguments. Found as 512 KiB moved per head in nanoLM's step: the softmax loop
/// stayed on the host while `sgemm` still took raw pointers, and each fresh
/// buffer was uploaded.
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
        assert!(stderr.contains("1048656 bytes uploaded") && stderr.contains(", 1 submits"), "{}: {stderr}", device.name);
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
/// reads). nanoLM's attention gradients were 3 to 15 times off.
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
    let run = |backend: &str, device: usize| {
        let output = Command::new(env!("CARGO_BIN_EXE_cleave"))
            .args(["--backend", backend, "--no-openmp", "--no-debug-info", "--run"])
            .arg(&source)
            .env("CLEAVE_VK_DEVICE", device.to_string())
            .env("CLEAVE_VK_VALIDATION", "1")
            .output()
            .expect("cannot run cleave");
        (String::from_utf8_lossy(&output.stdout).into_owned(), String::from_utf8_lossy(&output.stderr).into_owned())
    };
    let (cpu, _) = run("cpu", 0);
    let expected = numbers(&cpu);
    assert_eq!(expected.len(), 12, "cpu: {cpu}");
    for (index, device) in devices.iter().enumerate() {
        let (gpu, stderr) = run("vulkan", index);
        let got = numbers(&gpu);
        assert_eq!(got.len(), expected.len(), "{}: {gpu}
{stderr}", device.name);
        for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
            assert!((g - e).abs() <= 1e-4 * e.abs().max(1.0), "{}: value {i}: {g} on the GPU, {e} on the CPU", device.name);
        }
        assert!(!stderr.contains("cleave Vulkan validation"), "{}: {stderr}", device.name);
    }
}
