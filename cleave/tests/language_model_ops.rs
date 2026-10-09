//! The differentiable operations a character-level language model needs
//! (`doc/plan-nanolm.md`, step 2), each checked by its forward value and its
//! gradient: an `adjoint` rule with a non-differentiable (integer) parameter,
//! embeddings, cross-entropy on integer targets, GELU.

use cleave::pipeline::CodegenOptions;

/// Compiles `src` through the real pipeline and returns what its `main`
/// returns.
/// Compiles and runs `src`'s `main` (`fn main() -> f32`) through the real
/// pipeline (`cleave::run`, what `--run` uses): `main`'s four bytes read as
/// an `f32`. Tasks off: this binary's tests run in parallel (`doc/backlog.md`,
/// several programs spawning tasks at once in one process).
fn run(src: &str) -> f32 {
    let options = CodegenOptions { openmp: false, tasks: false, ..Default::default() };
    let bits = cleave::run::run_source("test.cleave", src, &options).unwrap_or_else(|e| panic!("{}", e.join("
")));
    f32::from_bits(bits as u32)
}

fn close(a: f32, b: f32) -> bool {
    (a - b).abs() <= 1e-4 * (1.0 + b.abs())
}

/// `_` in an `adjoint` rule: the parameter gets no contribution (an integer
/// one has no gradient). The rule still applies to the others.
#[test]
fn an_adjoint_rule_can_leave_an_integer_parameter_out() {
    let g = run("
        use convert;
        algebra Times<T> {
            fn times(x: T, k: i32) -> T;
            adjoint times(x, k), u: (times(u, k), _);
        }
        impl Times<f32> {
            fn times(x, k) { x * k.to() }
        }
        fn f(x: f32, k: i32) -> f32 { times(x, k) * x }
        df = grad(f, x);
        fn main() -> f32 { df(1.5, 3) }
    ");
    assert!(close(g, 9.0), "d(3x^2)/dx at 1.5 is 9, got {g}");
}

/// `embed(table, ids)` reads row `ids[n]` of the table for each `n`; its
/// gradient adds each upstream row back into the row it was read from (a row
/// read twice gets two contributions, a row never read none).
#[test]
fn an_embedding_reads_rows_and_its_gradient_adds_them_back() {
    let src = |out: &str| {
        format!(
            "
            use nn;
            fn loss(table: Tensor<f32, 3, 2>, ids: [i32; 3]) -> f32 {{ sum(embed(table, ids)) }}
            dtable = grad(loss, table);
            fn main() -> f32 {{
                let table = Tensor::<f32, 3, 2>(data: [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
                let ids: [i32; 3] = [2, 0, 2];
                let g = dtable(table, ids);
                {out}
            }}
        "
        )
    };
    assert_eq!(run(&src("loss(table, ids)")), 5.0 + 6.0 + 1.0 + 2.0 + 5.0 + 6.0);
    assert_eq!(run(&src("g[2, 1] * 100.0 + g[1, 0] * 10.0 + g[0, 0]")), 201.0);
}

/// `sparse_cross_entropy(z, y)`: the cross-entropy of each row of logits
/// against the class `y[b]`, summed over the batch, `-sum_b log
/// softmax(z)[b, y[b]]`; its gradient is `softmax(z)` minus one at the
/// target. Reference values computed in f64 (what PyTorch's
/// `F.cross_entropy(z, y, reduction="sum")` and its `.grad` give).
#[test]
fn cross_entropy_on_integer_targets_matches_pytorch() {
    let src = |out: &str| {
        format!(
            "
            use nn;
            fn loss(z: Tensor<f32, 2, 3>, y: [i32; 2]) -> f32 {{ sparse_cross_entropy(z, y) }}
            dz = grad(loss, z);
            fn main() -> f32 {{
                let z = Tensor::<f32, 2, 3>(data: [[1.0, 2.0, 0.5], [-1.0, 0.0, 3.0]]);
                let y: [i32; 2] = [1, 0];
                let g = dz(z, y);
                {out}
            }}
        "
        )
    };
    assert!(close(run(&src("loss(z, y)")), 4.5302527), "loss");
    assert!(close(run(&src("g[0, 0]")), 0.2312239), "g[0, 0]");
    assert!(close(run(&src("g[0, 1]")), -0.3714683), "g[0, 1]");
    assert!(close(run(&src("g[1, 0]")), -0.9828522), "g[1, 0]");
    assert!(close(run(&src("g[1, 2]")), 0.9362396), "g[1, 2]");
}

/// GELU, in the `tanh` form (`F.gelu(x, approximate="tanh")`): `0.5 x (1 +
/// tanh(sqrt(2/pi) (x + 0.044715 x^3)))`, and its derivative, at a few
/// points on both sides of zero. Reference values computed in f64.
#[test]
fn gelu_and_its_gradient_match_the_tanh_formula() {
    let src = |out: &str| {
        format!(
            "
            use nn;
            fn loss(x: Tensor<f32, 1, 4>) -> f32 {{ sum(gelu(x)) }}
            dx = grad(loss, x);
            fn main() -> f32 {{
                let x = Tensor::<f32, 1, 4>(data: [[-2.0, -0.5, 0.3, 1.7]]);
                let g = dx(x);
                {out}
            }}
        "
        )
    };
    assert!(close(run(&src("loss(x)")), 1.6097885), "loss");
    for (k, expected) in [-0.0860993f32, 0.1326301, 0.7322955, 1.1159146].into_iter().enumerate() {
        let got = run(&src(&format!("g[0, {k}]")));
        assert!(close(got, expected), "g[0, {k}]: expected {expected}, got {got}");
    }
}

/// A model with a single field (a bigram table) trains like any other: its
/// optimizer state is a tuple of one entry, not the entry itself (a
/// one-element comprehension with no target used to collapse to its
/// element, and `state[0]` then indexed into it).
#[test]
fn a_model_with_a_single_field_trains() {
    for opt in ["Sgd(lr: 0.25)", "Adam(lr: 0.25, beta1: 0.9, beta2: 0.999, eps: 0.00000001)"] {
        let got = run(&format!(
            "
            use nn;
            struct One {{ w: Tensor<f32, 1, 2> }}
            impl Trainable<One> {{}}
            fn main() -> f32 {{
                let opt = {opt};
                let m = One(w: Tensor::<f32, 1, 2>(data: [[1.0, 2.0]]));
                let g = One(w: Tensor::<f32, 1, 2>(data: [[1.0, 1.0]]));
                let (m2, s2) = step(opt, m, g, init_state(opt, m));
                m2.w[0, 0]
            }}
        "
        ));
        // Both take a step of exactly `lr` here (Adam's first step is
        // `lr * sign(g)`, up to `eps`).
        assert!(close(got, 0.75), "{opt}: expected 0.75, got {got}");
    }
}

/// One Adam step on a tensor built from a literal, gradient the tensor
/// itself: `1 - lr * sign(1)` everywhere. `doc/backlog.md` ("Adam on literal
/// tensors returns a wrong value after MLIR inlining") had it at ~0.9999
/// once the inputs were inlined constants, at this exact shape; it no longer
/// reproduces, bare or in a model.
#[test]
fn adam_on_a_literal_tensor_takes_its_step() {
    let bare = run("
        use nn;
        fn main() -> f32 {
            let opt = Adam(lr: 0.25, beta1: 0.9, beta2: 0.999, eps: 0.00000001);
            let w = Tensor::<f32, 8, 8>(data: [[1.0; 8]; 8]);
            let (w2, s2) = step(opt, w, w, init_state(opt, w));
            w2[3, 5]
        }
    ");
    assert!(close(bare, 0.75), "bare tensor: expected 0.75, got {bare}");
    let model = run("
        use nn;
        struct One { w: Tensor<f32, 8, 8> }
        impl Trainable<One> {}
        fn main() -> f32 {
            let opt = Adam(lr: 0.25, beta1: 0.9, beta2: 0.999, eps: 0.00000001);
            let m = One(w: Tensor::<f32, 8, 8>(data: [[1.0; 8]; 8]));
            let (m2, s2) = step(opt, m, m, init_state(opt, m));
            m2.w[3, 5]
        }
    ");
    assert!(close(model, 0.75), "model: expected 0.75, got {model}");
}

/// `layer_norm(x, g, b)`: each row normalized to mean 0 and variance 1 (`eps`
/// 1e-5, PyTorch's default), then scaled by `g` and shifted by `b`; its
/// gradient with respect to all three. The loss weights the output by a
/// fixed `w` (a plain sum would give `x` a zero gradient: a normalized row
/// always sums to the same thing). Reference values from PyTorch
/// (`F.layer_norm`, float64).
#[test]
fn layer_norm_and_its_gradients_match_pytorch() {
    let src = |out: &str| {
        format!(
            "
            use nn;
            fn loss(x: Tensor<f32, 2, 4>, g: Tensor<f32, 1, 4>, b: Tensor<f32, 1, 4>, w: Tensor<f32, 2, 4>) -> f32 {{
                sum(layer_norm(x, g, b) * w)
            }}
            dx = grad(loss, x);
            dg = grad(loss, g);
            db = grad(loss, b);
            fn main() -> f32 {{
                let x = Tensor::<f32, 2, 4>(data: [[1.0, 2.0, 3.0, 5.0], [-1.0, 0.0, 0.5, 4.0]]);
                let g = Tensor::<f32, 1, 4>(data: [[1.5, -0.5, 1.0, 2.0]]);
                let b = Tensor::<f32, 1, 4>(data: [[0.1, 0.2, -0.3, 0.0]]);
                let w = Tensor::<f32, 2, 4>(data: [[1.0, -2.0, 0.5, 3.0], [2.0, 1.0, -1.0, 0.5]]);
                {out}
            }}
        "
        )
    };
    let check = |expr: &str, expected: f32| {
        let got = run(&src(expr));
        assert!(close(got, expected), "{expr}: expected {expected}, got {got}");
    };
    check("loss(x, g, b, w)", 6.2842238);
    check("layer_norm(x, g, b)[1, 3]", 3.3186119);
    for (k, e) in [0.8789521f32, -0.2511337, -1.3812195, 0.7534011].into_iter().enumerate() {
        check(&format!("dx(x, g, b, w)[0, {k}]"), e);
    }
    for (k, e) in [1.1426554f32, -0.6526114, -0.8865225, 0.3964785].into_iter().enumerate() {
        check(&format!("dx(x, g, b, w)[1, {k}]"), e);
    }
    for (k, e) in [-3.1743804f32, 0.5495771, 0.2836319, 5.3934755].into_iter().enumerate() {
        check(&format!("dg(x, g, b, w)[0, {k}]"), e);
    }
    for (k, e) in [3.0f32, -1.0, -0.5, 3.5].into_iter().enumerate() {
        check(&format!("db(x, g, b, w)[0, {k}]"), e);
    }
}

/// `softmax(z)` row by row, and its gradient `y * (u - rowsum(u * y))`.
/// Reference values from PyTorch (`F.softmax(z, dim=1)`, float64).
#[test]
fn row_softmax_and_its_gradient_match_pytorch() {
    let src = |out: &str| {
        format!(
            "
            use nn;
            fn loss(z: Tensor<f32, 2, 3>, v: Tensor<f32, 2, 3>) -> f32 {{ sum(softmax(z) * v) }}
            dz = grad(loss, z);
            fn main() -> f32 {{
                let z = Tensor::<f32, 2, 3>(data: [[1.0, 2.0, 0.5], [-1.0, 0.0, 3.0]]);
                let v = Tensor::<f32, 2, 3>(data: [[1.0, -2.0, 0.5], [2.0, 1.0, -1.0]]);
                {out}
            }}
        "
        )
    };
    let check = |expr: &str, expected: f32| {
        let got = run(&src(expr));
        assert!(close(got, expected), "{expr}: expected {expected}, got {got}");
    };
    check("loss(z, v)", -1.8110486);
    check("softmax(z)[1, 2]", 0.9362396);
    for (k, e) in [0.4522086f32, -0.6563648, 0.2041562].into_iter().enumerate() {
        check(&format!("dz(z, v)[0, {k}]"), e);
    }
    for (k, e) in [0.0489627f32, 0.0864819, -0.1354446].into_iter().enumerate() {
        check(&format!("dz(z, v)[1, {k}]"), e);
    }
}

/// Causal multi-head attention on `[N, D]` matrices (`N` = sequences x
/// positions, `D` = heads x head width), `AttentionShape<L, DH>` giving the
/// sequence length and the head width: per sequence and head,
/// `softmax(mask(Q Kᵀ / sqrt(DH))) V`, a position attending to itself and
/// the ones before it. Here 2 sequences of 8 positions, 2 heads of 8. The
/// inputs are formulas both sides compute; each result is checked through a
/// fingerprint `sum(g * r)` covering every entry, and a few entries. Reference
/// values from PyTorch (float64, reshaped to `[B, H, L, DH]`).
#[test]
fn causal_attention_and_its_gradients_match_pytorch() {
    let src = |out: &str| {
        format!(
            "
            use nn;
            fn f(a: f32, b: f32, c: f32, s: f32) -> Tensor<f32, 16, 16> {{
                [for i in 0..16: [for j in 0..16: s * Transcendental::tanh(a * i.to() + b * j.to() + c)]]
            }}
            fn attend(q: Tensor<f32, 16, 16>, k: Tensor<f32, 16, 16>, v: Tensor<f32, 16, 16>) -> Tensor<f32, 16, 16> {{
                causal_attention(q, k, v, AttentionShape::<8, 8>())
            }}
            fn loss(q: Tensor<f32, 16, 16>, k: Tensor<f32, 16, 16>, v: Tensor<f32, 16, 16>, w: Tensor<f32, 16, 16>) -> f32 {{
                sum(attend(q, k, v) * w)
            }}
            dq = grad(loss, q);
            dk = grad(loss, k);
            dv = grad(loss, v);
            fn main() -> f32 {{
                let q = f(0.35, -0.6, 1.0, 1.5);
                let k = f(-0.3, 0.45, 1.0, 1.5);
                let v = f(0.25, -0.9, 0.5, 1.0);
                let w = f(0.9, -1.3, 0.3, 1.0);
                let r = f(-0.5, 0.8, 0.2, 1.0);
                {out}
            }}
        "
        )
    };
    let check = |expr: &str, expected: f32| {
        let got = run(&src(expr));
        assert!(close(got, expected), "{expr}: expected {expected}, got {got}");
    };
    check("loss(q, k, v, w)", 137.85068);
    check("sum(attend(q, k, v) * r)", -163.22666);
    check("attend(q, k, v)[1, 0]", 0.5425336);
    check("attend(q, k, v)[9, 2]", 0.6440305);
    check("sum(dq(q, k, v, w) * r)", 2.6630752);
    check("sum(dk(q, k, v, w) * r)", -4.0037066);
    check("sum(dv(q, k, v, w) * r)", -168.80259);
    check("dq(q, k, v, w)[9, 2]", -0.0122076);
    check("dk(q, k, v, w)[1, 0]", -0.1153489);
    check("dk(q, k, v, w)[9, 2]", 0.2420756);
    check("dv(q, k, v, w)[1, 0]", 2.0653594);
    check("dv(q, k, v, w)[9, 2]", 1.7124473);
}

/// A matmul whose column count isn't a multiple of the schedule's tile of 16
/// (104: the output layer over the alphabet) compiles through the real CLI
/// and computes the product, checked against a plain triple loop. The
/// partial last tile is padded; its write-back, a copy of dynamic size, used
/// to reach the affine pass and fail the whole compilation
/// (`cleave-lower-dynamic-copies`). Through the CLI because
/// this file's in-process pipeline doesn't run the matmul schedule.
#[test]
fn a_matmul_with_a_partial_column_tile_compiles_and_computes_the_product() {
    let dir = std::env::temp_dir().join("cleave-language-model-ops");
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join("partial_tile.cleave");
    std::fs::write(
        &source,
        "
        use nn;
        fn f(x: Tensor<f32, 64, 32>, w: Tensor<f32, 32, 104>) -> Tensor<f32, 64, 104> { matmul(x, w) }
        fn main() -> i32 {
            rand_seed(1);
            let x: Tensor<f32, 64, 32> = Init::he();
            let w: Tensor<f32, 32, 104> = Init::he();
            let y = f(x, w);
            let mut d = 0.0;
            for i in 0..64 { for j in 0..104 {
                let mut s = 0.0;
                for k in 0..32 { s = s + x[i, k] * w[k, j]; };
                d = d + (s - y[i, j]) * (s - y[i, j]);
            }; };
            if d < 0.0001 { 1 } else { 0 }
        }
        ",
    )
    .unwrap();
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_cleave"))
        .args(["--no-openmp", "--no-debug-info", "--run"])
        .arg(&source)
        .output()
        .expect("cannot run cleave");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stdout.contains("main returned: 1"), "stdout: {stdout}
stderr: {stderr}");
}

/// `matmul_transpose_b` compiled as a function of its own (`--no-inline`):
/// the affine super-vectorizer gives it permuted vector transfers (column
/// reads) that `--convert-vector-to-llvm` can't lower, which used to reach
/// LLVM's translation and crash it without a word. A second
/// `--convert-vector-to-scf` now lowers them, and only when such a transfer
/// exists, since it would otherwise turn every vector transfer into a scalar
/// loop (`cleave-lower-permuted-transfers`, `cleave-mlir`).
#[test]
fn a_matmul_with_a_transposed_operand_compiles_without_inlining() {
    let dir = std::env::temp_dir().join("cleave-language-model-ops");
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join("transpose_no_inline.cleave");
    std::fs::write(
        &source,
        "
        use nn;
        fn main() -> i32 {
            rand_seed(1);
            let q: Tensor<f32, 8, 16> = Init::he();
            let p: Tensor<f32, 8, 8> = matmul_transpose_b(q, q);
            let mut d = 0.0;
            for i in 0..8 { for j in 0..8 {
                let mut s = 0.0;
                for k in 0..16 { s = s + q[i, k] * q[j, k]; };
                d = d + (s - p[i, j]) * (s - p[i, j]);
            }; };
            if d < 0.0001 { 1 } else { 0 }
        }
        ",
    )
    .unwrap();
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_cleave"))
        .args(["--no-openmp", "--no-debug-info", "--no-inline", "--run"])
        .arg(&source)
        .output()
        .expect("cannot run cleave");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stdout.contains("main returned: 1"), "status {:?}\nstdout: {stdout}\nstderr: {stderr}", output.status);
}

/// Every matmul is vectorized, a column count that isn't a multiple of the
/// schedule's 16 included (104): the schedule peels the loop into whole tiles
/// plus a static remainder (`matmul_vectorize.transform.mlir`,
/// `@tile_peel_vectorize`). A matmul the schedule leaves behind still
/// computes the right result, through scalar loops, so only the IR can tell:
/// a silently failed peel once left every matmul of nanoLM's kernel scalar
/// (5x slower, every correctness test green). Checked on the IR after
/// bufferization, with the shapes of nanoLM's head and its gradients.
#[test]
fn matmuls_with_a_partial_tile_are_vectorized() {
    let dir = std::env::temp_dir().join("cleave-language-model-ops");
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join("vectorized.cleave");
    let dump = dir.join("vectorized_post_dealloc.mlir");
    let _ = std::fs::remove_file(&dump);
    std::fs::write(
        &source,
        "
        use nn;
        fn head(x: Tensor<f32, 64, 128>, w: Tensor<f32, 128, 104>) -> Tensor<f32, 64, 104> { matmul(x, w) }
        fn back(u: Tensor<f32, 64, 104>, w: Tensor<f32, 128, 104>) -> Tensor<f32, 64, 128> { matmul_transpose_b(u, w) }
        fn main() -> i32 {
            rand_seed(1);
            let x: Tensor<f32, 64, 128> = Init::he();
            let w: Tensor<f32, 128, 104> = Init::he();
            let y = head(x, w);
            let z = back(y, w);
            if z[0, 0] != 12345.0 { 1 } else { 0 }
        }
        ",
    )
    .unwrap();
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_cleave"))
        .args(["--no-openmp", "--no-debug-info", "--run"])
        .arg(&source)
        .env("CLEAVE_DUMP_POST_DEALLOC", &dump)
        .output()
        .expect("cannot run cleave");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("main returned: 1"), "stdout: {stdout}\nstderr: {}", String::from_utf8_lossy(&output.stderr));
    let ir = std::fs::read_to_string(&dump).expect("no IR dumped");
    assert!(!ir.contains("linalg.matmul"), "a matmul was left unvectorized");
    assert!(ir.contains("vector.outerproduct"), "no vectorized matmul at all");
}

/// An adjoint rule whose contributions are the parts of one call's tuple:
/// the e-graph shares the call (one node for the three reads), so the
/// backward is computed once — what a fused attention backward needs.
#[test]
fn an_adjoint_rule_can_take_its_contributions_from_one_tuple() {
    let src = |out: &str| {
        format!(
            "
            algebra Prod<T> {{
                fn prod(a: T, b: T) -> T;
                fn prod_back(a: T, b: T, u: T) -> (T, T);
                adjoint prod(a, b), u: (prod_back(a, b, u)[0], prod_back(a, b, u)[1]);
            }}
            impl Prod<f32> {{
                fn prod(a, b) {{ a * b }}
                fn prod_back(a, b, u) {{ (u * b, u * a) }}
            }}
            fn f(a: f32, b: f32) -> f32 {{ prod(a, b) * a }}
            da = grad(f, a);
            db = grad(f, b);
            fn main() -> f32 {{ {out} }}
        "
        )
    };
    // f = a^2 b: df/da = 2ab = 12 at (2, 3), df/db = a^2 = 4.
    assert!(close(run(&src("da(2.0, 3.0)")), 12.0));
    assert!(close(run(&src("db(2.0, 3.0)")), 4.0));
}

/// Tensors read out of a tuple a function returned are views of the
/// tuple's buffers, not references of their own: the tuple must outlive
/// every use of them. It used to be released right after the destructuring,
/// so `a` and `b` below pointed into freed memory by the time they were used
/// (found through a fused attention backward returning `(dq, dk, dv)`:
/// wrong gradients from the second transformer block on).
/// `refcount.rs::TensorViews`.
#[test]
fn tensors_destructured_from_a_returned_tuple_stay_valid() {
    let got = run("
        use nn;
        #[no_inline]
        fn two(x: Tensor<f32, 64, 64>) -> (Tensor<f32, 64, 64>, Tensor<f32, 64, 64>) { (x + x, x * x) }
        #[no_inline]
        fn churn(a: Tensor<f32, 64, 64>) -> Tensor<f32, 64, 64> {
            let mut s = a;
            for i in 0..20 { s = s + a; };
            s
        }
        fn main() -> f32 {
            rand_seed(5);
            let x: Tensor<f32, 64, 64> = Init::he();
            let (a, b) = two(x);
            let c = churn(a);
            let d1 = b[3, 4] - x[3, 4] * x[3, 4];
            let d2 = c[3, 4] - a[3, 4] * 21.0;
            if d1 * d1 < 0.000001 and d2 * d2 < 0.0001 { 1.0 } else { 0.0 }
        }
    ");
    assert_eq!(got, 1.0);
}

/// `#[no_inline]` is a property of the one method that declares it: Adam's
/// leaf `step` declares it, `Sgd`'s doesn't, and both are specializations of
/// `Optimizer::step`. It used to be read off whichever impl was being walked
/// when the specializations were collected, so the last impl's attribute
/// landed on every impl's specializations (MNIST's `Sgd` step went out of
/// line, a second slower). And a plain `fn` declaring it stays a function
/// even when its body is a chain of calls the e-graph pass would otherwise
/// walk through (nanoLM's transformer `block`).
#[test]
fn no_inline_applies_to_the_declaring_method_only() {
    let dir = std::env::temp_dir().join("cleave-language-model-ops");
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join("no_inline_per_impl.cleave");
    std::fs::write(
        &source,
        "
        use nn;
        algebra Bump<T> { fn bump(x: T) -> T; }
        impl<const R: i32, const C: i32> Bump<Tensor<f32, R, C>> {
            fn bump(x) { x * x }
        }
        impl<const N: i32> Bump<Tensor<f32, N>> {
            #[no_inline]
            fn bump(x) { x + x }
        }
        fn square(x: Tensor<f32, 4, 4>) -> Tensor<f32, 4, 4> { x * x }
        #[no_inline]
        fn chain(x: Tensor<f32, 4, 4>) -> Tensor<f32, 4, 4> { square(square(x)) }
        // An axiom applies here (`matmul(transpose(a), b)`), so the e-graph
        // pass rewrites this body, walking through what it calls.
        fn twice(x: Tensor<f32, 4, 4>) -> Tensor<f32, 4, 4> { matmul(transpose(chain(x)), x) }
        fn main() -> f32 {
            let a: Tensor<f32, 4, 4> = [for i in 0..4: [for j in 0..4: 2.0]];
            let b: Tensor<f32, 8> = [for i in 0..8: 3.0];
            bump(a)[1, 2] + bump(b)[5] + twice(a)[0, 0]
        }
        ",
    )
    .unwrap();
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_cleave"))
        .args(["--no-openmp", "--no-debug-info", "--dump-mlir-lowered"])
        .arg(&source)
        .output()
        .expect("cannot run cleave");
    let ir = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "stderr: {}", String::from_utf8_lossy(&output.stderr));
    let defined = |name: &str| ir.lines().any(|l| l.trim_start().starts_with("llvm.func") && l.contains(name));
    assert!(defined("@\"Bump::bump<Tensor<f32, 8>>\""), "the `#[no_inline]` method was inlined");
    assert!(!defined("@\"Bump::bump<Tensor<f32, 4, 4>>\""), "the other impl's method was kept out of line");
    assert!(defined("@chain"), "the `#[no_inline]` plain fn was inlined");
}

/// `tanh`/`exp`/`log` on tensors become polynomial approximations, vector
/// arithmetic, not intrinsics: LLVM has no vector math library here and
/// scalarized each `llvm.intr.tanh` on a vector into one libm `tanhf` call
/// per element — 26% of a nanoLM training step, the GELUs.
/// `pipeline.rs`, `cleave_mlir::approximate_math`.
#[test]
fn transcendentals_on_tensors_are_not_libm_calls() {
    let dir = std::env::temp_dir().join("cleave-language-model-ops");
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join("approximated_math.cleave");
    std::fs::write(
        &source,
        "
        use nn;
        fn act(x: Tensor<f32, 64, 64>) -> Tensor<f32, 64, 64> { gelu(x) + Transcendental::exp(x) }
        fn main() -> f32 {
            rand_seed(1);
            let x: Tensor<f32, 64, 64> = Init::he();
            act(x)[3, 4]
        }
        ",
    )
    .unwrap();
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_cleave"))
        .args(["--no-openmp", "--no-debug-info", "--dump-mlir-lowered"])
        .arg(&source)
        .output()
        .expect("cannot run cleave");
    let ir = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "stderr: {}", String::from_utf8_lossy(&output.stderr));
    assert!(ir.contains("llvm.func @act") || ir.contains("llvm.func @main"), "no IR dumped");
    for intrinsic in ["llvm.intr.tanh", "llvm.intr.exp"] {
        assert!(!ir.contains(intrinsic), "`{intrinsic}` left in the IR: it becomes one libm call per element");
    }
}

/// A BLAS helper hands `sgemm`'s output straight back: the buffer passed as
/// `c` is the function's result, no copy. `to_buffer` used to give the extern
/// a dynamic-layout view, and the identity-layout result type then forced
/// One-Shot Bufferize to allocate and copy the whole output on every call
/// (`mlir_lower.rs::build_to_buffer_dynamic_layout`).
#[test]
fn a_blas_helper_returns_its_output_without_copying_it() {
    let dir = std::env::temp_dir().join("cleave-language-model-ops");
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join("blas_no_copy.cleave");
    let dump = dir.join("blas_no_copy_post_dealloc.mlir");
    let _ = std::fs::remove_file(&dump);
    std::fs::write(
        &source,
        "
        use nn;
        fn main() -> i32 {
            rand_seed(1);
            let a: Tensor<f32, 64, 32> = Init::he();
            let b: Tensor<f32, 48, 32> = Init::he();
            let c = blas_matmul_transpose_b(a, b);
            if c[1, 2] == 12345.0 { 1 } else { 0 }
        }
        ",
    )
    .unwrap();
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_cleave"))
        .args(["--no-openmp", "--no-debug-info", "--run"])
        .arg(&source)
        .env("CLEAVE_DUMP_POST_DEALLOC", &dump)
        .output()
        .expect("cannot run cleave");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("main returned: 0"), "stdout: {stdout}\nstderr: {}", String::from_utf8_lossy(&output.stderr));
    let ir = std::fs::read_to_string(&dump).expect("no IR dumped");
    let helper: String = ir
        .lines()
        .skip_while(|l| !l.contains("func.func private @\"blas_matmul_transpose_b"))
        .take_while(|l| !l.starts_with("  }"))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(!helper.is_empty(), "the helper isn't in the IR");
    assert!(!helper.contains("memref.copy"), "the helper copies its output:\n{helper}");
}

/// Tensors handed over inside an aggregate are written in place, not copied
/// into it (`cleave-forward-dead-source-copies`,
/// `forward_out_param_copies`): an array filled by loops and returned in a
/// tuple (`Tensor(data: buf)`'s defensive copy dropped, the tuple element's
/// storage allocated before the loops), and a function's result stored in a
/// struct field (the call writes the field directly). Every whole-tensor copy
/// of a nanoLM training step was one of these.
#[test]
fn tensors_handed_over_in_aggregates_are_not_copied() {
    let dir = std::env::temp_dir().join("cleave-language-model-ops");
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join("aggregates_no_copy.cleave");
    let dump = dir.join("aggregates_no_copy_post_dealloc.mlir");
    let _ = std::fs::remove_file(&dump);
    std::fs::write(
        &source,
        "
        use nn;
        struct Pair { a: Tensor<f32, 64, 64>, b: Tensor<f32, 64, 64> }
        #[no_inline]
        fn filled(x: f32) -> (Tensor<f32, 64, 64>, Tensor<f32, 64, 64>) {
            let mut p: [[f32; 64]; 64] = mlir::memref::alloc();
            let mut q: [[f32; 64]; 64] = mlir::memref::alloc();
            for i in 0..64 { for j in 0..64 { p[i, j] = x; q[i, j] = x + 1.0; }; };
            (Tensor(data: p), Tensor(data: q))
        }
        #[no_inline]
        fn doubled(t: Tensor<f32, 64, 64>) -> Tensor<f32, 64, 64> { t + t }
        #[no_inline]
        fn pair(x: f32) -> Pair {
            let (p, q) = filled(x);
            Pair(a: doubled(p), b: doubled(q))
        }
        fn main() -> i32 {
            let r = pair(2.0);
            if r.a[3, 4] == 4.0 and r.b[5, 6] == 6.0 { 0 } else { 1 }
        }
        ",
    )
    .unwrap();
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_cleave"))
        .args(["--no-openmp", "--no-debug-info", "--run"])
        .arg(&source)
        .env("CLEAVE_DUMP_POST_DEALLOC", &dump)
        .output()
        .expect("cannot run cleave");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("main returned: 0"), "stdout: {stdout}\nstderr: {}", String::from_utf8_lossy(&output.stderr));
    let ir = std::fs::read_to_string(&dump).expect("no IR dumped");
    for name in ["@filled", "@pair"] {
        let body: String = ir
            .lines()
            .skip_while(|l| !(l.contains("func.func") && l.contains(name)))
            .take_while(|l| !l.starts_with("  }"))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(!body.is_empty(), "`{name}` isn't in the IR");
        let copies: Vec<&str> = body
            .lines()
            .filter(|l| {
                let Some(rest) = l.trim().strip_prefix("memref.copy ") else { return false };
                let operands: Vec<&str> = rest.split(" :").next().unwrap_or("").split(", ").collect();
                operands.len() == 2 && operands[0] != operands[1]
            })
            .collect();
        assert!(copies.is_empty(), "`{name}` copies a tensor:\n{}", copies.join("\n"));
    }
}

/// Nucleus sampling over weights 0.5, 0.3, 0.15, 0.05 (logits their logs):
/// `top_p` 0.8 keeps the first two tokens only, drawn in proportion 5:3;
/// `top_p` 1 keeps all four.
#[test]
fn nucleus_sampling_draws_only_inside_the_nucleus() {
    let src = |top_p: &str| {
        format!(
            "
            use nn;
            fn main() -> f32 {{
                rand_seed(3);
                let p: [f32; 4] = [0.5, 0.3, 0.15, 0.05];
                let z: Tensor<f32, 1, 4> = [for i in 0..1: [for j in 0..4: log(p[j])]];
                let mut counts: [f32; 4] = [0.0, 0.0, 0.0, 0.0];
                for i in 0..4000 {{
                    let c = sample_row_top_p(z, 0, 1.0, {top_p});
                    counts[c] = counts[c] + 1.0;
                }};
                // Outside the first two tokens, times 1000, plus the first's share.
                (counts[2] + counts[3]) * 1000.0 + counts[0] / 4000.0
            }}
            "
        )
    };
    let nucleus = run(&src("0.8"));
    assert!(nucleus < 1.0, "drew outside the nucleus: {nucleus}");
    assert!((nucleus - 0.625).abs() < 0.03, "first token's share {nucleus}, expected 5/8");
    let all = run(&src("1.0"));
    assert!(all >= 1000.0, "top_p 1 must keep the tail: {all}");
}

/// RMSNorm on the same inputs as `layer_norm_and_its_gradients_match_pytorch`
/// (no bias). Reference values from PyTorch (`F.rms_norm`, eps 1e-5, float64).
#[test]
fn rms_norm_and_its_gradients_match_pytorch() {
    let src = |out: &str| {
        format!(
            "
            use nn;
            fn loss(x: Tensor<f32, 2, 4>, g: Tensor<f32, 1, 4>, w: Tensor<f32, 2, 4>) -> f32 {{
                sum(rms_norm(x, g) * w)
            }}
            dx = grad(loss, x);
            dg = grad(loss, g);
            fn main() -> f32 {{
                let x = Tensor::<f32, 2, 4>(data: [[1.0, 2.0, 3.0, 5.0], [-1.0, 0.0, 0.5, 4.0]]);
                let g = Tensor::<f32, 1, 4>(data: [[1.5, -0.5, 1.0, 2.0]]);
                let w = Tensor::<f32, 2, 4>(data: [[1.0, -2.0, 0.5, 3.0], [2.0, 1.0, -1.0, 0.5]]);
                {out}
            }}
        "
        )
    };
    let check = |expr: &str, expected: f32| {
        let got = run(&src(expr));
        assert!(close(got, expected), "{expr}: expected {expected}, got {got}");
    };
    check("loss(x, g, w)", 11.4497364);
    check("rms_norm(x, g)[1, 3]", 3.8523428);
    for (k, e) in [0.1929752f32, -0.2545620, -0.7020991, 0.4844915].into_iter().enumerate() {
        check(&format!("dx(x, g, w)[0, {k}]"), e);
    }
    for (k, e) in [1.4585863f32, -0.2407714, -0.4885217, 0.4257119].into_iter().enumerate() {
        check(&format!("dx(x, g, w)[1, {k}]"), e);
    }
    for (k, e) in [-0.6428296f32, -1.2810246, 0.2396128, 5.7669279].into_iter().enumerate() {
        check(&format!("dg(x, g, w)[0, {k}]"), e);
    }
}

/// SiLU, `x * sigmoid(x)`, and its declared derivative. Reference values from
/// PyTorch (`F.silu`, float64).
#[test]
fn silu_and_its_gradient_match_pytorch() {
    let src = |out: &str| {
        format!(
            "
            use nn;
            fn loss(x: Tensor<f32, 1, 4>) -> f32 {{ sum(silu(x)) }}
            dx = grad(loss, x);
            fn main() -> f32 {{
                let x = Tensor::<f32, 1, 4>(data: [[-2.0, -0.5, 0.3, 1.7]]);
                let g = dx(x);
                {out}
            }}
        "
        )
    };
    assert!(close(run(&src("loss(x)")), 1.1825656), "loss");
    for (k, expected) in [-0.0907842f32, 0.2600388, 0.6477800, 1.0675645].into_iter().enumerate() {
        let got = run(&src(&format!("g[0, {k}]")));
        assert!(close(got, expected), "g[0, {k}]: expected {expected}, got {got}");
    }
}

/// A SwiGLU MLP, `down(silu(gate(x)) * up(x))`, and the gradients of a
/// weighted sum of its output with respect to its input and its three
/// layers. Reference values from PyTorch (float64).
#[test]
fn swiglu_and_its_gradients_match_pytorch() {
    let src = |out: &str| {
        format!(
            "
            use nn;
            fn loss(x: Tensor<f32, 2, 2>, m: SwiGlu<f32, 2, 3>, w: Tensor<f32, 2, 2>) -> f32 {{
                sum(m.swiglu_forward(x) * w)
            }}
            dx = grad(loss, x);
            dm = grad(loss, m);
            fn main() -> f32 {{
                let x = Tensor::<f32, 2, 2>(data: [[1.0, -0.5], [0.25, 2.0]]);
                let m = SwiGlu(
                    gate: Dense(w: Tensor::<f32, 2, 3>(data: [[0.5, -1.0, 0.25], [1.5, 0.75, -0.5]]), b: Tensor::<f32, 1, 3>(data: [[0.1, -0.2, 0.3]])),
                    up: Dense(w: Tensor::<f32, 2, 3>(data: [[-0.25, 0.5, 1.0], [0.75, -1.25, 0.5]]), b: Tensor::<f32, 1, 3>(data: [[0.0, 0.2, -0.1]])),
                    down: Dense(w: Tensor::<f32, 3, 2>(data: [[1.0, -0.5], [0.25, 0.75], [-1.0, 0.5]]), b: Tensor::<f32, 1, 2>(data: [[0.05, -0.05]]))
                );
                let w = Tensor::<f32, 2, 2>(data: [[1.0, -2.0], [0.5, 3.0]]);
                {out}
            }}
        "
        )
    };
    let check = |expr: &str, expected: f32| {
        let got = run(&src(expr));
        assert!(close(got, expected), "{expr}: expected {expected}, got {got}");
    };
    check("loss(x, m, w)", -8.8885515);
    for (k, e) in [-1.5327024f32, -1.2508525, 5.6281748, -10.8421485].into_iter().enumerate() {
        check(&format!("dx(x, m, w)[{}, {}]", k / 2, k % 2), e);
    }
    for (k, e) in [-0.9198546f32, -1.1304234, -1.0614750, -2.8402336, -9.7795097, 1.0233799].into_iter().enumerate() {
        check(&format!("dm(x, m, w).gate.w[{}, {}]", k / 3, k % 3), e);
    }
    for (k, e) in [-0.9141935f32, 0.7994769, -1.1590729, -6.1339943, 3.5257898, 0.1110694].into_iter().enumerate() {
        check(&format!("dm(x, m, w).up.w[{}, {}]", k / 3, k % 3), e);
    }
    for (k, e) in [2.2727056f32, 13.2893058, -1.2037814, -4.3594160, 0.2320251, -1.4781435].into_iter().enumerate() {
        check(&format!("dm(x, m, w).down.w[{}, {}]", k / 2, k % 2), e);
    }
}

/// `sin` and `cos` on tensors, composed, and the gradient derived from their
/// `Transcendental` rules. Reference values from PyTorch (float64).
#[test]
fn sin_and_cos_and_their_gradients_match_pytorch() {
    let src = |out: &str| {
        format!(
            "
            use nn;
            fn loss(x: Tensor<f32, 1, 4>) -> f32 {{ sum(sin(x) * cos(x + x)) }}
            dx = grad(loss, x);
            fn main() -> f32 {{
                let x = Tensor::<f32, 1, 4>(data: [[-2.0, -0.5, 0.3, 1.7]]);
                let g = dx(x);
                {out}
            }}
        "
        )
    };
    assert!(close(run(&src("loss(x)")), -0.3795147), "loss");
    for (k, expected) in [1.6483288f32, -0.3326855, 0.4547467, 0.6313889].into_iter().enumerate() {
        let got = run(&src(&format!("g[0, {k}]")));
        assert!(close(got, expected), "g[0, {k}]: expected {expected}, got {got}");
    }
}

/// RoPE over 2 sequences of 2 positions, 2 heads of 4 dimensions, and its
/// gradient (the inverse rotation). Reference values from PyTorch, Llama's
/// `rotate_half` convention (float64).
#[test]
fn rope_and_its_gradient_match_pytorch() {
    let src = |out: &str| {
        format!(
            "
            use nn;
            use convert;
            fn md(a: i32, m: i32) -> i32 {{ a - (a / m) * m }}
            fn loss(x: Tensor<f32, 4, 8>, w: Tensor<f32, 4, 8>) -> f32 {{
                sum(rope(x, AttentionShape::<2, 4>()) * w)
            }}
            dx = grad(loss, x);
            fn main() -> f32 {{
                let x: Tensor<f32, 4, 8> = [for r in 0..4: [for c in 0..8: 0.5 * md(r * 8 + c, 7).to() - 1.0]];
                let w: Tensor<f32, 4, 8> = [for r in 0..4: [for c in 0..8: 0.25 * md(r * 3 + c * 5, 9).to() - 1.0]];
                {out}
            }}
        "
        )
    };
    let check = |expr: &str, expected: f32| {
        let got = run(&src(expr));
        assert!(close(got, expected), "{expr}: expected {expected}, got {got}");
    };
    check("loss(x, w)", 3.4633817);
    let y = [-0.9920553f32, 0.9799503, 1.2311890, 2.0098998, -0.5403023, -0.5049749, -0.8414710, 0.4949751];
    for (k, e) in y.into_iter().enumerate() {
        check(&format!("rope(x, AttentionShape::<2, 4>())[3, {k}]"), e);
    }
    let dx = [-1.1714055f32, 0.2549874, 0.4362443, 0.4974750, -0.4805189, 0.7599623, 0.2856599, 0.9924501];
    for (k, e) in dx.into_iter().enumerate() {
        check(&format!("dx(x, w)[3, {k}]"), e);
    }
}

/// Three AdamW steps on one tensor with fixed gradients, against
/// `torch.optim.AdamW` (lr 0.1, betas 0.9 / 0.95, weight decay 0.1, float64).
#[test]
fn adamw_matches_pytorch() {
    let src = |k: usize| {
        format!(
            "
            use nn;
            fn main() -> f32 {{
                let opt = AdamW(lr: 0.1, beta1: 0.9, beta2: 0.95, eps: 0.00000001, weight_decay: 0.1);
                let mut w = Tensor::<f32, 1, 4>(data: [[1.0, -2.0, 0.5, 3.0]]);
                let mut s = init_state(opt, w);
                (w, s) = step(opt, w, Tensor::<f32, 1, 4>(data: [[0.5, -1.0, 0.25, 2.0]]), s);
                (w, s) = step(opt, w, Tensor::<f32, 1, 4>(data: [[-0.3, 0.8, 1.5, -0.5]]), s);
                (w, s) = step(opt, w, Tensor::<f32, 1, 4>(data: [[0.1, 0.2, -0.7, 1.0]]), s);
                w[0, {k}]
            }}
        "
        )
    };
    for (k, e) in [0.8273726f32, -1.8423232, 0.2721164, 2.7044603].into_iter().enumerate() {
        let got = run(&src(k));
        assert!(close(got, e), "w[0, {k}]: expected {e}, got {got}");
    }
}

/// Clipping a `Dense` gradient of global norm 4.62 to 2, against
/// `torch.nn.utils.clip_grad_norm_` (float64).
#[test]
fn clip_grad_norm_matches_pytorch() {
    let src = |out: &str| {
        format!(
            "
            use nn;
            fn main() -> f32 {{
                let g = Dense(
                    w: Tensor::<f32, 2, 3>(data: [[1.0, -2.0, 0.5], [3.0, 0.25, -1.5]]),
                    b: Tensor::<f32, 1, 3>(data: [[0.5, -0.75, 2.0]])
                );
                let c = clip_grad_norm(g, 2.0);
                {out}
            }}
        "
        )
    };
    let check = |expr: &str, expected: f32| {
        let got = run(&src(expr));
        assert!(close(got, expected), "{expr}: expected {expected}, got {got}");
    };
    check("mlir::math::sqrt(squared_norm(g))", 4.6233105);
    for (k, e) in [0.4325904f32, -0.8651807, 0.2162952, 1.2977711, 0.1081476, -0.6488855].into_iter().enumerate() {
        check(&format!("c.w[{}, {}]", k / 3, k % 3), e);
    }
    for (k, e) in [0.2162952f32, -0.3244428, 0.8651807].into_iter().enumerate() {
        check(&format!("c.b[0, {k}]"), e);
    }
    // Under the threshold, unchanged.
    check("clip_grad_norm(g, 10.0).w[1, 0]", 3.0);
}

/// Two Muon steps on a wide (3 x 4) and on a tall (4 x 3) matrix, against
/// Keller Jordan's algorithm written in PyTorch (Newton-Schulz quintic, Nesterov
/// momentum, `sqrt(max(1, R / C))` scale, decoupled decay; float64).
#[test]
fn muon_matches_its_reference() {
    let src = |shape: &str, w: &str, g1: &str, g2: &str, out: &str| {
        format!(
            "
            use nn;
            fn main() -> f32 {{
                let opt = Muon(lr: 0.05, momentum: 0.95, weight_decay: 0.01,
                               adamw: AdamW(lr: 0.001, beta1: 0.9, beta2: 0.95, eps: 0.00000001, weight_decay: 0.0));
                let mut w = Tensor::<f32, {shape}>(data: {w});
                let mut s = init_state(opt, w);
                (w, s) = step(opt, w, Tensor::<f32, {shape}>(data: {g1}), s);
                (w, s) = step(opt, w, Tensor::<f32, {shape}>(data: {g2}), s);
                {out}
            }}
        "
        )
    };
    let wide = "[[1.0, -2.0, 0.5, 3.0], [0.25, 1.5, -1.0, 0.5], [2.0, 0.0, -0.5, 1.0]]";
    let g1 = "[[0.5, -1.0, 0.25, 2.0], [-0.3, 0.8, 1.5, -0.5], [0.1, 0.2, -0.7, 1.0]]";
    let g2 = "[[1.0, 0.5, -0.25, 0.0], [0.4, -0.6, 0.2, 1.2], [-1.0, 0.3, 0.9, -0.2]]";
    let expected_wide = [
        0.9542974f32, -1.9868207, 0.4994867, 2.9233893, 0.2463944, 1.4934879, -1.0631486, 0.4836372, 2.0267163,
        -0.0457288, -0.4877112, 0.9588599,
    ];
    for (k, e) in expected_wide.into_iter().enumerate() {
        let got = run(&src("3, 4", wide, g1, g2, &format!("w[{}, {}]", k / 4, k % 4)));
        assert!(close(got, e), "wide w[{}, {}]: expected {e}, got {got}", k / 4, k % 4);
    }
    // The transposes: Newton-Schulz on the wide orientation, the update `sqrt(4/3)` larger.
    let tall = "[[1.0, 0.25, 2.0], [-2.0, 1.5, 0.0], [0.5, -1.0, -0.5], [3.0, 0.5, 1.0]]";
    let t1 = "[[0.5, -0.3, 0.1], [-1.0, 0.8, 0.2], [0.25, 1.5, -0.7], [2.0, -0.5, 1.0]]";
    let t2 = "[[1.0, 0.4, -1.0], [0.5, -0.6, 0.3], [-0.25, 0.2, 0.9], [0.0, 1.2, -0.2]]";
    let expected_tall = [
        0.9473819f32, 0.2458752, 2.0311586, -1.9850912, 1.4927125, -0.0528031, 0.4994846, -1.0730723, -0.4858874,
        2.9120016, 0.4811832, 0.9526502,
    ];
    for (k, e) in expected_tall.into_iter().enumerate() {
        let got = run(&src("4, 3", tall, t1, t2, &format!("w[{}, {}]", k / 3, k % 3)));
        assert!(close(got, e), "tall w[{}, {}]: expected {e}, got {got}", k / 3, k % 3);
    }
}

/// An `Embedding` used at both ends (its rows read for a batch of ids, then
/// scored against every row, a tied output head): its gradient is the sum of
/// both uses. Reference values from PyTorch (float64).
#[test]
fn a_tied_embedding_gets_the_gradient_of_both_uses() {
    let src = |out: &str| {
        format!(
            "
            use nn;
            use convert;
            fn md(a: i32, m: i32) -> i32 {{ a - (a / m) * m }}
            fn loss(e: Embedding<f32, 5, 3>, w: Tensor<f32, 4, 5>) -> f32 {{
                let ids: [i32; 4] = [2, 0, 4, 2];
                sum(embedding_logits(e, embedding_forward(e, ids)) * w)
            }}
            de = grad(loss, e);
            fn main() -> f32 {{
                let e = Embedding(table: Tensor::<f32, 5, 3>(data: [[0.5, -1.0, 0.25], [1.5, 0.75, -0.5], [-0.25, 0.5, 1.0], [0.75, -1.25, 0.5], [0.1, 0.2, -0.3]]));
                let w: Tensor<f32, 4, 5> = [for r in 0..4: [for c in 0..5: 0.25 * md(r * 3 + c * 5, 9).to() - 1.0]];
                {out}
            }}
        "
        )
    };
    let check = |expr: &str, expected: f32| {
        let got = run(&src(expr));
        assert!(close(got, expected), "{expr}: expected {expected}, got {got}");
    };
    check("loss(e, w)", -4.1006250);
    let expected = [
        1.075f32, 1.65, -3.35, 0.325, -0.85, 0.9, 1.225, -0.425, -3.175, -0.775, 1.45, 0.825, -0.3, -0.5375, -0.5375,
    ];
    for (k, e) in expected.into_iter().enumerate() {
        check(&format!("de(e, w).table[{}, {}]", k / 3, k % 3), e);
    }
}

/// A model mixing an `Embedding` and a `Dense`, stepped by `Muon` as a whole:
/// the table goes to AdamW, the `Dense` matrix to Muon, its bias (a row of
/// one) to AdamW — each equal to the same optimizer applied to the part alone.
#[test]
fn muon_routes_each_part_of_a_model_to_its_update() {
    let src = |out: &str| {
        format!(
            "
            use nn;
            struct Net {{ e: Embedding<f32, 4, 3>, d: Dense<f32, 3, 2> }}
            impl Trainable<Net> {{}}
            fn main() -> f32 {{
                rand_seed(2);
                let adamw = AdamW(lr: 0.01, beta1: 0.9, beta2: 0.95, eps: 0.00000001, weight_decay: 0.1);
                let opt = Muon(lr: 0.05, momentum: 0.95, weight_decay: 0.01, adamw: adamw);
                let m = Net(e: Init::xavier(), d: Init::xavier());
                let g = Net(e: Init::xavier(), d: Dense(w: Init::xavier(), b: Init::xavier()));
                let (whole, _) = step(opt, m, g, init_state(opt, m));
                let (table, _) = step(adamw, m.e.table, g.e.table, init_state(adamw, m.e.table));
                let (w, _) = step(opt, m.d.w, g.d.w, init_state(opt, m.d.w));
                let (b, _) = step(adamw, m.d.b, g.d.b, init_state(adamw, m.d.b));
                {out}
            }}
        "
        )
    };
    for (whole, part) in [("whole.e.table[3, 1]", "table[3, 1]"), ("whole.d.w[2, 0]", "w[2, 0]"), ("whole.d.b[0, 1]", "b[0, 1]")] {
        let got = run(&src(&format!("{whole} - {part}")));
        assert!(got.abs() < 1e-7, "{whole} differs from {part} by {got}");
        let moved = run(&src(&format!("{whole} - m.{}", &whole[6..])));
        assert!(moved.abs() > 1e-6, "{whole} didn't move");
    }
    // Muon's matrix update differs from AdamW's on the same matrix.
    let differs = run(&src("let (aw, _) = step(adamw, m.d.w, g.d.w, init_state(adamw, m.d.w)); whole.d.w[2, 0] - aw[2, 0]"));
    assert!(differs.abs() > 1e-6, "the Dense matrix was stepped by AdamW, not Muon");
}

/// `sum` over a large matrix, and its gradient (`Sum::broadcast`): an
/// array-repeat literal there was converted to CPS one element at a time,
/// recursively, and an embedding table's size (4096 x 384) overflowed the
/// compiler's stack. A `splat` now.
#[test]
fn the_gradient_of_a_sum_over_a_large_matrix_compiles() {
    let src = "
        use nn;
        fn loss(t: Tensor<f32, 4096, 384>) -> f32 { sum(t * t) }
        dt = grad(loss, t);
        fn main() -> f32 {
            let t: Tensor<f32, 4096, 384> = Ring::one();
            dt(t)[4095, 383] + loss(t) / 1000000.0
        }
    ";
    // d(sum(t²))/dt = 2t = 2, plus 4096 * 384 / 1e6.
    let got = run(src);
    assert!(close(got, 2.0 + 1.572864), "got {got}");
}

/// A literal-count array repeat the size of an embedding table, nested
/// (`[[u; 384]; 4096]`): one node, filled by a loop (`cps.rs::
/// fill_array_repeat`); it used to become 1.5 million copies of `u` and
/// overflow the compiler's stack.
#[test]
fn a_large_nested_array_repeat_is_filled_by_a_loop() {
    let got = run("
        use nn;
        fn main() -> f32 {
            let u: f32 = 2.5;
            let a: [[f32; 384]; 4096] = [[u; 384]; 4096];
            let mut s = 0.0;
            for i in 0..4096 { s = s + a[i, 383]; };
            s
        }
    ");
    assert!(close(got, 10240.0), "got {got}");
}

/// A comprehension over a one-field struct's fields, summed by a loop over
/// its parts, as over a two-field one: the one-element result used to be a
/// tuple of one (`__Tuple1<f32>`), which only a folded index reaches, so the
/// loop failed with `no impl Index<__Tuple1<f32>, _>`. An array of one now,
/// like two or more (`stdlib/nn`'s `GradNorm` over an `Embedding`).
#[test]
fn a_comprehension_over_one_field_is_indexed_like_over_two() {
    let got = run("
        struct One { a: f32 }
        struct Two { a: f32, b: f32 }
        fn sum1(m: One) -> f32 {
            let parts = [for i in 0..m.len(): m[i] * 2.0];
            let mut t = 0.0;
            for i in 0..parts.len() { t = t + parts[i]; };
            t
        }
        fn sum2(m: Two) -> f32 {
            let parts = [for i in 0..m.len(): m[i] * 2.0];
            let mut t = 0.0;
            for i in 0..parts.len() { t = t + parts[i]; };
            t
        }
        fn main() -> f32 { sum1(One(a: 1.5)) + sum2(Two(a: 1.0, b: 2.0)) }
    ");
    assert!(close(got, 9.0), "got {got}");
}


/// A light struct at or above the by-pointer threshold (128 bytes,
/// `mlir_lower.rs::BY_POINTER_MIN_BYTES`; three tensors' descriptors here)
/// passed to a function that isn't inlined. By value, LLVM expanded it into
/// its scalars, stored below the stack pointer with no probe: past Windows'
/// guard page an access violation, depending on the order of the stores and
/// on how much of the stack the thread had committed (nanoLM v2's
/// `Optimizer::step`, 58 KB of arguments; not reproducible on demand in a
/// small program). So what is checked is the ABI itself: the function
/// receives pointers to copies (`mlir_lower.rs::by_pointer`), and still
/// computes the right value. Size doesn't change the ABI past the
/// threshold, so the struct stays small: one of ~57 KB took 80 s to compile.
#[test]
fn a_large_light_struct_crosses_a_call_by_pointer() {
    let src = "
        use nn;
        struct P { a: Tensor<f32, 1, 1>, b: Tensor<f32, 1, 1>, c: Tensor<f32, 1, 1> }
        fn mk(v: f32) -> P {
            P(a: [for r in 0..1: [for c in 0..1: v]], b: [for r in 0..1: [for c in 0..1: v]], c: [for r in 0..1: [for c in 0..1: v]])
        }
        #[no_inline]
        fn read(x: P, y: P) -> f32 { x.a[0, 0] + y.c[0, 0] }
        fn main() -> f32 { read(mk(1.5), mk(2.0)) }
    ";
    let got = run(src);
    assert!(close(got, 3.5), "got {got}");
    let dir = std::env::temp_dir().join("cleave-language-model-ops");
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join("large_light_struct.cleave");
    std::fs::write(&source, src).unwrap();
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_cleave"))
        .args(["--no-openmp", "--no-debug-info", "--dump-mlir-lowered"])
        .arg(&source)
        .output()
        .expect("cannot run cleave");
    let lowered = String::from_utf8_lossy(&output.stdout);
    let signature = lowered
        .lines()
        .find(|l| l.contains("llvm.func @read("))
        .unwrap_or_else(|| panic!("no `read` in the lowered module:
{}", String::from_utf8_lossy(&output.stderr)));
    assert!(
        signature.contains("@read(%arg0: !llvm.ptr, %arg1: !llvm.ptr)"),
        "both structs must arrive by pointer: {}...",
        &signature[..signature.len().min(160)]
    );
}

/// Every call passing a large light struct by pointer has its own argument
/// slot (`mlir_lower.rs::entry_alloca`), in its function's entry block.
/// Without `cleave_mlir::hoist_arg_slots`, an inlined call's slots land
/// wherever the call was (inlining `relay` into `main`'s loop: a dynamic
/// allocation per iteration), and all of a function's slots add up in its
/// frame: nanoLM v2's `train_gpt` reached 1.4 MB, past the main thread's
/// stack. With it, every slot is back in the entry block, and an ordinary
/// call's is live only around the call (`lifetime.start`/`lifetime.end`), so
/// that LLVM's stack coloring shares storage between them.
#[test]
fn argument_slots_sit_in_the_entry_block_with_bounded_lifetimes() {
    let src = "
        use nn;
        struct P { a: Tensor<f32, 1, 1>, b: Tensor<f32, 1, 1>, c: Tensor<f32, 1, 1> }
        fn mk(v: f32) -> P {
            P(a: [for r in 0..1: [for c in 0..1: v]], b: [for r in 0..1: [for c in 0..1: v]], c: [for r in 0..1: [for c in 0..1: v]])
        }
        #[no_inline]
        fn read(x: P, y: P, s: f32) -> f32 { s + x.a[0, 0] + x.b[0, 0] + x.c[0, 0] + y.a[0, 0] + y.b[0, 0] + y.c[0, 0] }
        fn relay(x: P, y: P, s: f32) -> f32 { read(x, y, s) }
        fn main() -> f32 {
            // Built here, not returned by `mk`: a struct returned by pointer
            // is passed on from where it lives, with no slot of its own.
            let x = P(a: [for r in 0..1: [for c in 0..1: 0.5]], b: [for r in 0..1: [for c in 0..1: 0.5]], c: [for r in 0..1: [for c in 0..1: 0.5]]);
            let y = P(a: [for r in 0..1: [for c in 0..1: 1.0]], b: [for r in 0..1: [for c in 0..1: 1.0]], c: [for r in 0..1: [for c in 0..1: 1.0]]);
            let mut s = read(x, y, 0.0);
            for i in 0..99 { s = relay(x, y, s); };
            s
        }
    ";
    let got = run(src);
    assert!(close(got, 450.0), "got {got}");
    let dir = std::env::temp_dir().join("cleave-language-model-ops");
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join("argument_slots.cleave");
    let object = dir.join("argument_slots.obj");
    let dump = dir.join("argument_slots.mlir");
    std::fs::write(&source, src).unwrap();
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_cleave"))
        .args(["--no-openmp", "--no-debug-info", "--emit-object"])
        .arg(&object)
        .arg(&source)
        .env("CLEAVE_DUMP_LLVM_DIALECT", &dump)
        .output()
        .expect("cannot run cleave");
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    let module = std::fs::read_to_string(&dump).unwrap();
    assert!(!module.contains("arg_slot"), "the slots' marks must be dropped");
    let start = module.find("llvm.func @main(").expect("no `main` in the module");
    let end = module[start + 1..].find("llvm.func ").map_or(module.len(), |e| start + 1 + e);
    let main = &module[start..end];
    let entry_end = main.find("^bb").unwrap_or(main.len());
    let allocas = main.matches("llvm.alloca").count();
    assert!(allocas >= 4, "`main` should hold its calls' slots: {allocas} allocas");
    assert_eq!(main[entry_end..].matches("llvm.alloca").count(), 0, "every slot in `main`'s entry block");
    // `main`'s own call to `read`, `relay`'s inlined one, and `main`'s to
    // `relay` (inlined: stored, then loaded): two slots each.
    let starts = main.matches("llvm.intr.lifetime.start").count();
    assert_eq!(starts, main.matches("llvm.intr.lifetime.end").count());
    assert!(starts >= 4, "each ordinary call's slots get a lifetime: {starts}");
}

/// Every impl defines every function its algebra declares
/// (`pipeline.rs::check_impl_completeness`); the ones the check found
/// missing, filled in: scalar GELU/SiLU and their derivatives, a tensor's
/// `tanh`/`log`, `Complex`'s `zero`/`one`, and a rank-1 `Sum`'s
/// `broadcast`, which `sum`'s adjoint calls (differentiating `sum(x * x)`
/// on a rank-1 tensor panicked in the e-graph).
#[test]
fn the_functions_missing_from_their_impls_compute_what_they_should() {
    let scalar = run("
        use nn;
        fn main() -> f32 {
            let x: f32 = 1.0;
            // 0.841192, 1.082964, 0.731059, 0.927671
            gelu(x) * 1000.0 + gelu_grad(x) * 100.0 + silu(x) * 10.0 + silu_grad(x)
        }
    ");
    let expected = 0.841192 * 1000.0 + 1.082964 * 100.0 + 0.731059 * 10.0 + 0.927671;
    assert!((scalar - expected).abs() < 0.01, "scalar activations: expected {expected}, got {scalar}");
    let tensor = run("
        use nn;
        fn main() -> f32 {
            let t = Tensor::<f32, 2>(data: [0.5, 2.0]);
            Transcendental::tanh(t)[0] * 10.0 + log(t)[1]
        }
    ");
    let expected = 0.5f32.tanh() * 10.0 + 2.0f32.ln();
    assert!(close(tensor, expected), "tensor tanh/log: expected {expected}, got {tensor}");
    let complex = run("
        use complex;
        fn main() -> f32 {
            let o: Complex<f32> = Ring::one();
            let z: Complex<f32> = Ring::zero();
            o.real * 10.0 + o.imag + z.real + z.imag
        }
    ");
    assert!(close(complex, 10.0), "Complex one/zero: got {complex}");
    let grad_of_sum = run("
        use nn;
        fn f(x: Tensor<f32, 3>) -> f32 { sum(x * x) }
        df = grad(f, x);
        fn main() -> f32 { df(Tensor::<f32, 3>(data: [1.0, 2.0, 3.0]))[2] }
    ");
    assert!(close(grad_of_sum, 6.0), "d sum(x^2)/dx at 3: expected 6, got {grad_of_sum}");
}

/// An impl that leaves out one of its algebra's functions is an error at
/// the impl, naming what's missing, rather than a failure wherever the
/// missing function happens to be called.
#[test]
fn an_impl_missing_a_function_of_its_algebra_is_an_error() {
    let dir = std::env::temp_dir().join("cleave-language-model-ops");
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join("incomplete_impl.cleave");
    std::fs::write(
        &source,
        "
        algebra Shape<T> { fn area(x: T) -> f32; fn perimeter(x: T) -> f32; }
        struct Sq { side: f32 }
        impl Shape<Sq> { fn area(x) { x.side * x.side } }
        fn main() -> i32 { 0 }
        ",
    )
    .unwrap();
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_cleave"))
        .args(["--no-openmp", "--run"])
        .arg(&source)
        .output()
        .expect("cannot run cleave");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success(), "an incomplete impl must not compile");
    assert!(
        stderr.contains("`impl Shape<Sq>` doesn't define `perimeter`"),
        "the error should name the impl and the missing function: {stderr}"
    );
}

/// A generic impl method whose body doesn't type-check, whatever its
/// generics: the error is reported in the body, where it is. The method used
/// to be dropped silently instead, and every call to it then failed
/// elsewhere ("CPS: could not resolve call to `Twice::twice`").
#[test]
fn a_generic_impl_body_that_fails_is_an_error_in_the_body() {
    let dir = std::env::temp_dir().join("cleave-language-model-ops");
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join("failing_generic_impl.cleave");
    std::fs::write(
        &source,
        "algebra Twice<T> { fn twice(x: T) -> T; }\nimpl<T: Float> Twice<T> { fn twice(x) { x + true } }\nfn main() -> f32 { twice(1.5) }\n",
    )
    .unwrap();
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_cleave"))
        .args(["--no-openmp", "--run"])
        .arg(&source)
        .output()
        .expect("cannot run cleave");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success(), "the program must not compile");
    assert!(
        stderr.contains("failing_generic_impl.cleave:2:41") && !stderr.contains("could not resolve call"),
        "the error should be located at `x + true`, line 2, not at the call: {stderr}"
    );
}

/// Runs `source` through the CLI, expecting it to fail; its `stderr`.
fn cli_error(name: &str, source: &str) -> String {
    let dir = std::env::temp_dir().join("cleave-language-model-ops");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(name);
    std::fs::write(&path, source).unwrap();
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_cleave"))
        .args(["--no-openmp", "--run"])
        .arg(&path)
        .output()
        .expect("cannot run cleave");
    assert!(!output.status.success(), "{name} must not compile");
    String::from_utf8_lossy(&output.stderr).into_owned()
}

/// A call to a name nothing declares in scope (`x.to()` without `use
/// convert;`) is an error at the call, naming it, rather than a placeholder
/// type reported later as "could not be fully determined
/// (<unresolved-call:convert>)" among knock-on mismatches.
#[test]
fn a_call_to_nothing_in_scope_is_an_error_at_the_call() {
    let stderr = cli_error(
        "unknown_callee.cleave",
        "fn main() -> f32 {\n    let n: i32 = 3;\n    let x: f32 = n.to();\n    x\n}\n",
    );
    assert!(
        stderr.contains("unknown_callee.cleave:3:18")
            && stderr.contains("no `fn` or algebra method named `convert` is in scope")
            && !stderr.contains("unresolved-call"),
        "{stderr}"
    );
}

/// A generic the call leaves undetermined is reported at the call, where it
/// should be pinned, not in the callee's body where it is first needed.
#[test]
fn an_undetermined_generic_is_reported_at_the_call_that_leaves_it_open() {
    let stderr = cli_error(
        "undetermined_generic.cleave",
        "use convert;\nfn half<T: Float>() -> T {\n    let one: i32 = 1;\n    one.to() / 2.0\n}\nfn main() -> i32 {\n    let h = half();\n    0\n}\n",
    );
    assert!(
        stderr.contains("undetermined_generic.cleave:7:13") && stderr.contains("ambiguous dispatch"),
        "the error should be at `half()`, line 7: {stderr}"
    );
}

/// `CLEAVE_ALLOC_STATS=1` counts every allocation and prints the totals at
/// exit: the measure of what a program materializes. Two tensor temporaries
/// per iteration here, ten iterations: at least twenty allocations of 4 KiB.
#[test]
fn alloc_stats_count_what_a_program_materializes() {
    let dir = std::env::temp_dir().join("cleave-language-model-ops");
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join("alloc_stats.cleave");
    std::fs::write(
        &source,
        "
        use nn;
        fn main() -> f32 {
            rand_seed(1);
            let mut x: Tensor<f32, 32, 32> = Init::he();
            for i in 0..10 {
                x = (x + x) * x;
            };
            x[1, 2]
        }
        ",
    )
    .unwrap();
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_cleave"))
        .args(["--no-openmp", "--no-debug-info", "--run"])
        .arg(&source)
        .env("CLEAVE_ALLOC_STATS", "1")
        .output()
        .expect("cannot run cleave");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "stderr: {stderr}");
    let summary = stderr
        .lines()
        .find_map(|l| l.strip_prefix("CLEAVE_ALLOC_STATS: "))
        .unwrap_or_else(|| panic!("no allocation report: {stderr}"));
    let count: u64 = summary.split(' ').next().unwrap().parse().unwrap();
    assert!(count >= 20, "{summary}");
}

/// `slice(x, r0, c0)` reads the block of `x` at rows `r0..` and columns
/// `c0..`, its size the result type's; `update(x, part, r0, c0)` is `x`
/// with that block replaced, everything else untouched.
#[test]
fn a_slice_reads_a_block_and_an_update_replaces_it() {
    let src = |out: &str| {
        format!(
            "
            use nn;
            fn main() -> f32 {{
                let x = Tensor::<f32, 3, 4>(data: [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]);
                let s: Tensor<f32, 2, 2> = slice(x, 1, 2);
                let y = update(x, s + s, 0, 1);
                {out}
            }}
        "
        )
    };
    // `s` is [[7, 8], [11, 12]].
    assert_eq!(run(&src("s[0, 0] * 1000.0 + s[0, 1] * 100.0 + s[1, 0] * 10.0 + s[1, 1]")), 7000.0 + 800.0 + 110.0 + 12.0);
    // `y` is `x` with [[14, 16], [22, 24]] at rows 0..2, columns 1..3.
    assert_eq!(run(&src("y[0, 0] + y[0, 1] * 10.0 + y[1, 2] * 100.0 + y[2, 3] * 1000.0")), 1.0 + 140.0 + 2400.0 + 12000.0);
}

/// The adjoints: the gradient of a block read is that block's upstream
/// gradient placed in zeros; through an update, the overwritten block of
/// `x` gets nothing and `part` gets the block of the upstream gradient.
#[test]
fn slices_and_updates_have_gradients() {
    let src = |out: &str| {
        format!(
            "
            use nn;
            fn read(x: Tensor<f32, 3, 4>) -> f32 {{
                let s: Tensor<f32, 2, 2> = slice(x, 1, 2);
                sum(s * s)
            }}
            dread = grad(read, x);
            fn write(x: Tensor<f32, 3, 4>, p: Tensor<f32, 2, 2>) -> f32 {{
                let y = update(x, p * p, 0, 1);
                sum(y * x)
            }}
            dwrite_x = grad(write, x);
            dwrite_p = grad(write, p);
            fn main() -> f32 {{
                let x = Tensor::<f32, 3, 4>(data: [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]);
                let p = Tensor::<f32, 2, 2>(data: [[1.0, 2.0], [3.0, 4.0]]);
                let gr = dread(x);
                let gx = dwrite_x(x, p);
                let gp = dwrite_p(x, p);
                {out}
            }}
        "
        )
    };
    // d/dx sum(s²) = 2x on the block, 0 elsewhere.
    assert_eq!(run(&src("gr[1, 2] + gr[2, 3] * 10.0 + gr[0, 0] * 100.0 + gr[1, 1] * 1000.0")), 14.0 + 240.0);
    // `sum(y * x)`: d/dx = y + (dy/dx)ᵀ x, and `y` doesn't depend on `x` on
    // the block: there, `y` = p², elsewhere 2x.
    assert_eq!(run(&src("gx[0, 1] + gx[0, 0] * 10.0 + gx[2, 3] * 100.0")), 1.0 + 20.0 + 2400.0);
    // d/dp sum(p² ⊙ x[block]) = 2 p ⊙ x[block], the block at rows 0..2, columns 1..3.
    assert_eq!(run(&src("gp[0, 0] + gp[1, 1] * 10.0")), 2.0 * 1.0 * 2.0 + 10.0 * 2.0 * 4.0 * 7.0);
}

/// Elementwise arithmetic straight on a slice (a strided view after
/// bufferization) compiles without diagnostics: `--affine-super-vectorize`
/// can't vectorize a loop over a non-identity layout and said so as an
/// `error:` on a compilation that succeeded.
#[test]
fn arithmetic_on_a_slice_compiles_silently() {
    let dir = std::env::temp_dir().join("cleave-language-model-ops");
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join("slice_arithmetic.cleave");
    std::fs::write(
        &source,
        "
        use nn;
        fn main() -> f32 {
            let x = Tensor::<f32, 3, 4>(data: [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]);
            let s: Tensor<f32, 2, 2> = slice(x, 1, 2);
            let y = update(x, s + s, 0, 1);
            y[0, 1]
        }
        ",
    )
    .unwrap();
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_cleave"))
        .args(["--no-openmp", "--no-debug-info", "--run"])
        .arg(&source)
        .output()
        .expect("cannot run cleave");
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "stderr: {stderr}");
    assert!(stdout.contains(&format!("main returned: {}", 14.0f32.to_bits() as i32)), "stdout: {stdout}");
    assert!(!stderr.contains("error"), "diagnostics on a successful compilation: {stderr}");
}

/// Per-head products as attention computes them: each block of `q` and `k`
/// read as a slice, multiplied by `sgemm` straight into a slice of `out`,
/// put back by `update`. Right values, and no block is materialized: BLAS
/// reads the views with their real leading dimension and writes the
/// product where it belongs (`Sgemm::sgemm`, `stdlib/blas`).
#[test]
fn sgemm_reads_and_writes_slices_in_place() {
    let dir = std::env::temp_dir().join("cleave-language-model-ops");
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join("sgemm_slices.cleave");
    let dump = dir.join("sgemm_slices_post_dealloc.mlir");
    let _ = std::fs::remove_file(&dump);
    std::fs::write(
        &source,
        "
        use nn;
        #[no_inline]
        fn heads(q: Tensor<f32, 8, 6>, k: Tensor<f32, 8, 6>) -> Tensor<f32, 8, 16> {
            let mut out: Tensor<f32, 8, 16> = zero();
            for h in 0..2 {
                let qh: Tensor<f32, 8, 3> = slice(q, 0, h * 3);
                let kh: Tensor<f32, 8, 3> = slice(k, 0, h * 3);
                let dest: Tensor<f32, 8, 8> = slice(out, 0, h * 8);
                out = update(out, sgemm(false, true, 1.0, qh, kh, 0.0, dest), 0, h * 8);
            };
            out
        }
        fn main() -> i32 {
            rand_seed(1);
            let q: Tensor<f32, 8, 6> = Init::he();
            let k: Tensor<f32, 8, 6> = Init::he();
            let out = heads(q, k);
            let mut wrong = 0;
            for h in 0..2 {
                for i in 0..8 {
                    for j in 0..8 {
                        let mut s: f32 = 0.0;
                        for c in 0..3 { s = s + q[i, h * 3 + c] * k[j, h * 3 + c]; };
                        let d = out[i, h * 8 + j] - s;
                        if d * d > 0.000001 { wrong = wrong + 1; };
                    };
                };
            };
            wrong
        }
        ",
    )
    .unwrap();
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_cleave"))
        .args(["--no-openmp", "--no-debug-info", "--run"])
        .arg(&source)
        .env("CLEAVE_DUMP_POST_DEALLOC", &dump)
        .output()
        .expect("cannot run cleave");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stdout.contains("main returned: 0"), "stdout: {stdout}\nstderr: {stderr}");
    let ir = std::fs::read_to_string(&dump).expect("no IR dumped");
    let heads: String = ir
        .lines()
        .skip_while(|l| !l.contains("func.func private @heads"))
        .take_while(|l| !l.starts_with("  }"))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(!heads.is_empty(), "`heads` isn't in the IR");
    for block in ["memref<8x3xf32", "memref<8x8xf32"] {
        assert!(
            !heads.lines().any(|l| (l.contains("memref.alloc") || l.contains("memref.copy")) && l.contains(block)),
            "a block is copied:\n{heads}"
        );
    }
    // The loop builds `out` in the caller's buffer
    // (`cleave_mlir::forward_copies_to_destinations`): the one copy left
    // is `zero()`'s, its initialization.
    assert_eq!(heads.matches("memref.copy").count(), 1, "`out` is copied:\n{heads}");
}

/// A block written by `sgemm` while `sgemm` also reads it (`a` and the
/// destination are the same slice of `out`): the write can't go in place,
/// BLAS would overwrite what it is still reading. The copy stays, the
/// values are right.
#[test]
fn a_block_read_by_its_own_write_is_still_copied() {
    let src = "
        use nn;
        fn main() -> f32 {
            let x = Tensor::<f32, 2, 4>(data: [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]);
            let b = Tensor::<f32, 2, 2>(data: [[1.0, 1.0], [1.0, 2.0]]);
            let blk: Tensor<f32, 2, 2> = slice(x, 0, 2);
            let y = update(x, sgemm(false, false, 1.0, blk, b, 0.0, blk), 0, 2);
            // [[3, 4], [7, 8]] @ [[1, 1], [1, 2]] = [[7, 11], [15, 23]]
            y[0, 2] + y[0, 3] * 10.0 + y[1, 2] * 100.0 + y[1, 3] * 1000.0 + y[0, 0] * 10000.0
        }
    ";
    assert_eq!(run(src), 7.0 + 110.0 + 1500.0 + 23000.0 + 10000.0);
}

/// A dense layer and its activation on the BLAS tier (`BLAS_MIN_WORK`
/// lowered to 0): the activation is tiled by rows with the product fused in,
/// each tile's bias broadcast and `sgemm` (accumulating into it, `beta = 1`)
/// done in a scratch tile the activation then reads while it is in cache
/// (`cleave_mlir::blas_tile_and_fuse`, `lower_blas_matmuls`). The values
/// are the `linalg` tier's, up to the order of the sums; the only buffer the
/// product's size is the result.
#[test]
fn a_blas_product_is_computed_tile_by_tile_with_its_consumer() {
    let dir = std::env::temp_dir().join("cleave-language-model-ops");
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join("blas_tiles.cleave");
    let dump = dir.join("blas_tiles_post_dealloc.mlir");
    std::fs::write(
        &source,
        "
        use nn;
        fn layer(x: Tensor<f32, 512, 32>, w: Tensor<f32, 32, 48>, b: Tensor<f32, 1, 48>) -> Tensor<f32, 512, 48> {
            silu(matmul(x, w) + broadcast0(b))
        }
        fn main() -> f32 {
            rand_seed(1);
            let x: Tensor<f32, 512, 32> = Init::he();
            let w: Tensor<f32, 32, 48> = Init::he();
            let b: Tensor<f32, 1, 48> = Init::he();
            sum(layer(x, w, b))
        }
        ",
    )
    .unwrap();
    let run = |blas: bool| {
        let _ = std::fs::remove_file(&dump);
        let mut command = std::process::Command::new(env!("CARGO_BIN_EXE_cleave"));
        command.args(["--no-openmp", "--no-debug-info", "--run"]);
        if blas {
            command.args(["--define", "BLAS_MIN_WORK=0"]).env("CLEAVE_DUMP_POST_DEALLOC", &dump);
        }
        let output = command.arg(&source).output().expect("cannot run cleave");
        // `--run` exits with what `main` returns: the bits, not a status.
        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let bits: i32 = stdout
            .trim()
            .strip_prefix("main returned: ")
            .and_then(|b| b.parse().ok())
            .unwrap_or_else(|| panic!("stdout: {stdout}\nstderr: {}", String::from_utf8_lossy(&output.stderr)));
        f32::from_bits(bits as u32)
    };
    let (native, blas) = (run(false), run(true));
    assert!((native - blas).abs() <= 1e-4 * native.abs().max(1.0), "linalg {native}, BLAS {blas}");
    let ir = std::fs::read_to_string(&dump).expect("no IR dumped");
    let main: String = ir
        .lines()
        .skip_while(|l| !l.contains("func.func @main"))
        .take_while(|l| !l.starts_with("  }"))
        .collect::<Vec<_>>()
        .join("\n");
    let calls: Vec<usize> = main.lines().enumerate().filter(|(_, l)| l.contains("call @cleave_blas_sgemm")).map(|(i, _)| i).collect();
    let tile_loop = main.lines().position(|l| l.contains("scf.for")).expect("no tile loop");
    assert_eq!(calls.len(), 1, "one `sgemm`, per tile:\n{main}");
    assert!(calls[0] > tile_loop, "the `sgemm` isn't in the tile loop:\n{main}");
    assert!(main.contains("memref<128x48xf32>"), "no scratch tile:\n{main}");
    let whole = main.lines().filter(|l| l.contains("memref.alloc") && l.contains("memref<512x48xf32>")).count();
    assert_eq!(whole, 1, "a product-sized buffer besides the result:\n{main}");
}

/// Three dense layers on the BLAS tier in one function: each product is
/// computed by a tile loop carrying its output buffer, and each layer's
/// result is freed once the next has read it, not at the function's end
/// (`cleave_mlir::fold_passthrough_iter_args`: the loop's result is
/// the buffer it was given, so the deallocation's alias analysis frees each
/// buffer on its own instead of all together after a run-time alias check;
/// `dealloc_at_last_use` then frees each right after its last use).
#[test]
fn buffers_are_freed_after_their_last_use_through_tile_loops() {
    let dir = std::env::temp_dir().join("cleave-language-model-ops");
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join("last_use.cleave");
    let dump = dir.join("last_use_post_dealloc.mlir");
    std::fs::write(
        &source,
        "
        use nn;
        fn layer(x: Tensor<f32, 512, 32>, w: Tensor<f32, 32, 32>, b: Tensor<f32, 1, 32>) -> Tensor<f32, 512, 32> {
            silu(matmul(x, w) + broadcast0(b))
        }
        fn main() -> f32 {
            rand_seed(1);
            let x: Tensor<f32, 512, 32> = Init::he();
            let w: Tensor<f32, 32, 32> = Init::he();
            let b: Tensor<f32, 1, 32> = Init::he();
            sum(layer(layer(layer(x, w, b), w, b), w, b))
        }
        ",
    )
    .unwrap();
    let run = |blas: bool| {
        let _ = std::fs::remove_file(&dump);
        let mut command = std::process::Command::new(env!("CARGO_BIN_EXE_cleave"));
        command.args(["--no-openmp", "--no-debug-info", "--run"]);
        if blas {
            command.args(["--define", "BLAS_MIN_WORK=0"]).env("CLEAVE_DUMP_POST_DEALLOC", &dump);
        }
        let output = command.arg(&source).output().expect("cannot run cleave");
        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let bits: i32 = stdout
            .trim()
            .strip_prefix("main returned: ")
            .and_then(|b| b.parse().ok())
            .unwrap_or_else(|| panic!("stdout: {stdout}
stderr: {}", String::from_utf8_lossy(&output.stderr)));
        f32::from_bits(bits as u32)
    };
    let (native, blas) = (run(false), run(true));
    assert!((native - blas).abs() <= 1e-4 * native.abs().max(1.0), "linalg {native}, BLAS {blas}");
    let ir = std::fs::read_to_string(&dump).expect("no IR dumped");
    let main: Vec<&str> = ir
        .lines()
        .skip_while(|l| !l.contains("func.func @main"))
        .take_while(|l| !l.starts_with("  }"))
        .collect();
    let text = main.join("
");
    assert!(!text.contains("dealloc_helper"), "a run-time alias check before freeing:
{text}");
    let whole = |op: &str| -> Vec<usize> {
        main.iter().enumerate().filter(|(_, l)| l.contains(op) && l.contains("memref<512x32xf32>")).map(|(i, _)| i).collect()
    };
    let (allocs, deallocs) = (whole("memref.alloc"), whole("memref.dealloc"));
    assert!(allocs.len() >= 3 && !deallocs.is_empty(), "allocs {allocs:?}, deallocs {deallocs:?}:
{text}");
    assert!(
        deallocs[0] < *allocs.last().unwrap(),
        "every layer-sized buffer is freed after the last is allocated:
{text}"
    );
}

/// An elementwise op consuming two BLAS products (SwiGLU's `silu(x Wg) *
/// x Wu`): both fused into the one tile loop (it used to be tiled once per
/// product, the second time after the first had replaced it, and crash).
/// The values are the `linalg` tier's, up to the order of the sums.
#[test]
fn an_elementwise_op_of_two_blas_products_fuses_both() {
    let dir = std::env::temp_dir().join("cleave-language-model-ops");
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join("blas_two_products.cleave");
    std::fs::write(
        &source,
        "
        use nn;
        fn main() -> f32 {
            rand_seed(1);
            let x: Tensor<f32, 256, 32> = Init::he();
            let g: Tensor<f32, 32, 48> = Init::he();
            let u: Tensor<f32, 32, 48> = Init::he();
            sum(silu(matmul(x, g)) * matmul(x, u))
        }
        ",
    )
    .unwrap();
    let run = |blas: bool| {
        let mut command = std::process::Command::new(env!("CARGO_BIN_EXE_cleave"));
        command.args(["--no-openmp", "--no-debug-info", "--run"]);
        if blas {
            command.args(["--define", "BLAS_MIN_WORK=0"]);
        }
        let output = command.arg(&source).output().expect("cannot run cleave");
        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let bits: i32 = stdout
            .trim()
            .strip_prefix("main returned: ")
            .and_then(|b| b.parse().ok())
            .unwrap_or_else(|| panic!("stdout: {stdout}\nstderr: {}", String::from_utf8_lossy(&output.stderr)));
        f32::from_bits(bits as u32)
    };
    let (native, blas) = (run(false), run(true));
    assert!((native - blas).abs() <= 1e-4 * native.abs().max(1.0), "linalg {native}, BLAS {blas}");
}

/// An elementwise op whose operand is a tensor computed just for it (a
/// call's result, not fused into the op) writes its result into that
/// operand's buffer instead of a new one (`cleave_mlir::
/// reuse_dying_inputs`); an operand read again afterwards is left alone.
#[test]
fn an_elementwise_op_writes_into_an_operand_that_dies_there() {
    let dir = std::env::temp_dir().join("cleave-language-model-ops");
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join("reuse_dying.cleave");
    let dump = dir.join("reuse_dying_post_dealloc.mlir");
    let _ = std::fs::remove_file(&dump);
    std::fs::write(
        &source,
        "
        use nn;
        #[no_inline]
        fn twice(a: Tensor<f32, 64, 64>) -> Tensor<f32, 64, 64> { a + a }
        #[no_inline]
        fn dies(a: Tensor<f32, 64, 64>, b: Tensor<f32, 64, 64>) -> f32 {
            let t = twice(a);
            sum(t * b)
        }
        #[no_inline]
        fn lives(a: Tensor<f32, 64, 64>, b: Tensor<f32, 64, 64>) -> f32 {
            let t = twice(a);
            sum(t * b) + sum(t)
        }
        fn main() -> f32 {
            rand_seed(1);
            let a: Tensor<f32, 64, 64> = Init::he();
            let b: Tensor<f32, 64, 64> = Init::he();
            dies(a, b) + lives(a, b)
        }
        ",
    )
    .unwrap();
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_cleave"))
        .args(["--no-openmp", "--no-debug-info", "--run"])
        .arg(&source)
        .env("CLEAVE_DUMP_POST_DEALLOC", &dump)
        .output()
        .expect("cannot run cleave");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("main returned"), "stdout: {stdout}\nstderr: {}", String::from_utf8_lossy(&output.stderr));
    let ir = std::fs::read_to_string(&dump).expect("no IR dumped");
    let body = |name: &str| -> String {
        ir.lines()
            .skip_while(|l| !l.contains(&format!("func.func private @{name}(")))
            .take_while(|l| !l.starts_with("  }"))
            .collect::<Vec<_>>()
            .join("\n")
    };
    let allocs = |b: &str| b.lines().filter(|l| l.contains("memref.alloc") && l.contains("memref<64x64xf32>")).count();
    // `twice`'s result, plus the product's own buffer only where `t` lives on.
    assert_eq!(allocs(&body("dies")), 1, "`t * b` isn't written into `t`:\n{}", body("dies"));
    assert_eq!(allocs(&body("lives")), 2, "`t`, still read, was overwritten:\n{}", body("lives"));
}

/// A BLAS product (its buffer laid out dynamically for `sgemm`) passed to a
/// function, whose parameter has the plain layout: One-Shot Bufferize copies
/// it into a fresh buffer of the plain layout; the copy's destination becomes
/// the product's own buffer, of that same type
/// (`cleave_mlir::forward_copies_to_destinations`): no copy.
#[test]
fn a_blas_product_passed_to_a_function_is_not_copied() {
    let dir = std::env::temp_dir().join("cleave-language-model-ops");
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join("blas_to_call.cleave");
    let dump = dir.join("blas_to_call_post_dealloc.mlir");
    let _ = std::fs::remove_file(&dump);
    std::fs::write(
        &source,
        "
        use nn;
        #[no_inline]
        fn consume(t: Tensor<f32, 64, 48>) -> f32 { sum(t * t) }
        #[no_inline]
        fn produce(x: Tensor<f32, 64, 32>, w: Tensor<f32, 32, 48>, b: Tensor<f32, 1, 48>) -> f32 {
            consume(matmul(x, w) + broadcast0(b))
        }
        fn main() -> f32 {
            rand_seed(1);
            let x: Tensor<f32, 64, 32> = Init::he();
            let w: Tensor<f32, 32, 48> = Init::he();
            let b: Tensor<f32, 1, 48> = Init::he();
            produce(x, w, b)
        }
        ",
    )
    .unwrap();
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_cleave"))
        .args(["--no-openmp", "--no-debug-info", "--define", "BLAS_MIN_WORK=0", "--run"])
        .arg(&source)
        .env("CLEAVE_DUMP_POST_DEALLOC", &dump)
        .output()
        .expect("cannot run cleave");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("main returned"), "stdout: {stdout}\nstderr: {}", String::from_utf8_lossy(&output.stderr));
    let ir = std::fs::read_to_string(&dump).expect("no IR dumped");
    let produce: String = ir
        .lines()
        .skip_while(|l| !l.contains("func.func private @produce("))
        .take_while(|l| !l.starts_with("  }"))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(produce.contains("cleave_blas_sgemm"), "not on the BLAS tier:\n{produce}");
    assert!(!produce.contains("memref.copy"), "the product is copied:\n{produce}");
}

/// A model whose layers are an array (`layers: [Layer; 3]`, the depth a
/// constant) differentiates like the same model with three named fields:
/// the loop over the layers unrolls, each element read is a projection like
/// a field read, and the gradient has the model's shape, an array of layer
/// gradients. Same initialization, same values to the bit
/// (`doc/plan-struct-arrays.md`).
#[test]
fn a_model_whose_layers_are_an_array_has_the_gradient_of_named_layers() {
    let common = "
        use nn;
        define LAYERS: i32 = 3;
        struct Layer { d: Dense<f32, 16, 16> }
        impl Trainable<Layer> {}
        fn new_layer() -> Layer { Layer(d: Init::xavier()) }
    ";
    let array = |out: &str| {
        format!(
            "{common}
            struct Net {{ layers: [Layer; LAYERS], out: Dense<f32, 16, 4> }}
            impl Trainable<Net> {{}}
            // The loop in a function `loss` calls, as a model's forward is
            // written: its body unrolls inside `loss`'s gradient.
            fn forward(x: Tensor<f32, 8, 16>, net: Net) -> Tensor<f32, 8, 16> {{
                let mut h = x;
                for i in 0..LAYERS {{ h = relu(net.layers[i].d.dense_forward(h)); }};
                h
            }}
            fn loss(x: Tensor<f32, 8, 16>, net: Net) -> f32 {{
                sum(net.out.dense_forward(forward(x, net)))
            }}
            net_grad = grad(loss, net);
            fn main() -> f32 {{
                rand_seed(1);
                let a = new_layer();
                let b = new_layer();
                let c = new_layer();
                let net = Net(layers: [a, b, c], out: Init::xavier());
                let x: Tensor<f32, 8, 16> = Init::he();
                let g = net_grad(x, net);
                {out}
            }}"
        )
    };
    let named = |out: &str| {
        format!(
            "{common}
            struct Net {{ l1: Layer, l2: Layer, l3: Layer, out: Dense<f32, 16, 4> }}
            impl Trainable<Net> {{}}
            fn loss(x: Tensor<f32, 8, 16>, net: Net) -> f32 {{
                let h1 = relu(net.l1.d.dense_forward(x));
                let h2 = relu(net.l2.d.dense_forward(h1));
                let h3 = relu(net.l3.d.dense_forward(h2));
                sum(net.out.dense_forward(h3))
            }}
            net_grad = grad(loss, net);
            fn main() -> f32 {{
                rand_seed(1);
                let a = new_layer();
                let b = new_layer();
                let c = new_layer();
                let net = Net(l1: a, l2: b, l3: c, out: Init::xavier());
                let x: Tensor<f32, 8, 16> = Init::he();
                let g = net_grad(x, net);
                {out}
            }}"
        )
    };
    for (by_index, by_name) in [
        ("sum(g.layers[0].d.w)", "sum(g.l1.d.w)"),
        ("sum(g.layers[1].d.b)", "sum(g.l2.d.b)"),
        ("g.layers[2].d.w[3, 5]", "g.l3.d.w[3, 5]"),
        ("sum(g.out.w)", "sum(g.out.w)"),
    ] {
        let (a, n) = (run(&array(by_index)), run(&named(by_name)));
        assert_eq!(a.to_bits(), n.to_bits(), "{by_index}: {a} against {n}");
    }
}

/// A call `grad` can't see through (a branch on a run-time value in the
/// callee) is a located error naming it, not a gradient silently built from
/// what came before the call (it used to take a variable read at the call
/// for the function's result: `out`'s gradient came out the integer `1`, and
/// MLIR lowering crashed on it).
#[test]
fn grad_through_an_opaque_call_is_an_error_naming_it() {
    let dir = std::env::temp_dir().join("cleave-language-model-ops");
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join("grad_opaque_call.cleave");
    std::fs::write(
        &source,
        "
        use nn;
        struct Net { d: Dense<f32, 16, 16>, out: Dense<f32, 16, 4> }
        fn forward(x: Tensor<f32, 8, 16>, net: Net, k: i32) -> Tensor<f32, 8, 16> {
            if k > 2 { relu(net.d.dense_forward(x)) } else { net.d.dense_forward(x) }
        }
        fn loss(x: Tensor<f32, 8, 16>, net: Net, k: i32) -> f32 {
            sum(net.out.dense_forward(forward(x, net, k)))
        }
        net_grad = grad(loss, net);
        fn main() -> f32 {
            rand_seed(1);
            let net = Net(d: Init::xavier(), out: Init::xavier());
            let x: Tensor<f32, 8, 16> = Init::he();
            sum(net_grad(x, net, 3).d.w)
        }
        ",
    )
    .unwrap();
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_cleave"))
        .args(["--no-openmp", "--no-debug-info", "--run"])
        .arg(&source)
        .output()
        .expect("cannot run cleave");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("Assertion failed") && !stderr.contains("panicked"), "a crash, not an error:\n{stderr}");
    assert!(stderr.contains("net_grad") && stderr.contains("forward"), "the error doesn't name the call:\n{stderr}");
}

/// Decoding with a key/value cache (`rope_at`, `cached_attention`) computes
/// what `causal_attention` does over the whole sequence: tokens fed one at a
/// time, each one's rotated key and value written to its slot, its query
/// attending to the slots filled so far. With a cache of 4 slots on a
/// sequence of 8, the cache slides as a ring (the newest token in the oldest
/// slot, rotated by its absolute position): token 6 sees tokens 3..=6, as
/// `causal_attention` over those four alone, rotated from position 0, does.
#[test]
fn cached_attention_decodes_as_causal_attention_does() {
    let src = |out: &str| {
        format!(
            "
            use nn;
            use convert;
            fn f(a: f32, b: f32, c: f32) -> Tensor<f32, 8, 16> {{
                [for i in 0..8: [for j in 0..16: 1.5 * Transcendental::tanh(a * i.to() + b * j.to() + c)]]
            }}
            fn row(x: Tensor<f32, 8, 16>, t: i32) -> Tensor<f32, 1, 16> {{ slice(x, t, 0) }}
            // Token `t`'s output, the cache `L` slots long.
            fn decode<const L: i32>(q: Tensor<f32, 8, 16>, k: Tensor<f32, 8, 16>, v: Tensor<f32, 8, 16>, t: i32) -> Tensor<f32, 1, 16> {{
                let shape = AttentionShape::<L, 8>();
                let mut kc: Tensor<f32, L, 16> = uninitialized();
                let mut vc: Tensor<f32, L, 16> = uninitialized();
                let mut out: Tensor<f32, 1, 16> = uninitialized();
                for s in 0..t + 1 {{
                    let slot = s - (s / L) * L;
                    kc = update(kc, rope_at(row(k, s), s, shape), slot, 0);
                    vc = update(vc, row(v, s), slot, 0);
                    let n = if s + 1 < L {{ s + 1 }} else {{ L }};
                    out = cached_attention(rope_at(row(q, s), s, shape), kc, vc, n, shape);
                }};
                out
            }}
            fn full(q: Tensor<f32, 8, 16>, k: Tensor<f32, 8, 16>, v: Tensor<f32, 8, 16>) -> Tensor<f32, 8, 16> {{
                let shape = AttentionShape::<8, 8>();
                causal_attention(rope(q, shape), rope(k, shape), v, shape)
            }}
            // Rows 3..=6 alone, as one sequence of 4.
            fn w(x: Tensor<f32, 8, 16>) -> Tensor<f32, 4, 16> {{ slice(x, 3, 0) }}
            fn window(q: Tensor<f32, 8, 16>, k: Tensor<f32, 8, 16>, v: Tensor<f32, 8, 16>) -> Tensor<f32, 4, 16> {{
                let shape = AttentionShape::<4, 8>();
                causal_attention(rope(w(q), shape), rope(w(k), shape), w(v), shape)
            }}
            fn gap(a: Tensor<f32, 1, 16>, b: Tensor<f32, 1, 16>) -> f32 {{
                let mut m = 0.0;
                for c in 0..16 {{ let d = a[0, c] - b[0, c]; if d > m {{ m = d; }}; if 0.0 - d > m {{ m = 0.0 - d; }}; }};
                m
            }}
            fn main() -> f32 {{
                let q = f(0.35, -0.6, 1.0);
                let k = f(-0.3, 0.45, 1.0);
                let v = f(0.25, -0.9, 0.5);
                {out}
            }}
        "
        )
    };
    let check = |expr: &str| {
        let got = run(&src(expr));
        assert!(got < 1e-5, "{expr}: off by {got}");
    };
    for t in [0, 1, 5, 7] {
        check(&format!("gap(decode::<8>(q, k, v, {t}), row(full(q, k, v), {t}))"));
    }
    check("gap(decode::<4>(q, k, v, 6), slice(window(q, k, v), 3, 0))");
    check("gap(decode::<4>(q, k, v, 3), row(full(q, k, v), 3))");
}

/// A small GPT decoding with key/value caches (an array of tensors per
/// layer, `rope_at`, `cached_attention`) gives the logits its whole-sequence
/// forward pass gives, token by token: nanoLM's `generate` in miniature.
#[test]
fn a_gpt_decoding_with_caches_gives_its_forward_logits() {
    let src = |out: &str| {
        format!(
            "
            use nn;
            use convert;
            struct Blk {{
                n1: Tensor<f32, 1, 16>,
                wq: Dense<f32, 16, 16>, wk: Dense<f32, 16, 16>, wv: Dense<f32, 16, 16>, wo: Dense<f32, 16, 16>,
                n2: Tensor<f32, 1, 16>,
                mlp: SwiGlu<f32, 16, 32>
            }}
            struct G {{ tok: Embedding<f32, 12, 16>, blocks: [Blk; 2], nf: Tensor<f32, 1, 16> }}
            fn ones() -> Tensor<f32, 1, 16> {{ [for i in 0..1: [for j in 0..16: 1.0]] }}
            fn mk() -> Blk {{ Blk(n1: ones(), wq: Init::xavier(), wk: Init::xavier(), wv: Init::xavier(), wo: Init::xavier(), n2: ones(), mlp: Init::xavier()) }}
            #[no_inline]
            fn block<const N: i32>(x: Tensor<f32, N, 16>, b: Blk) -> Tensor<f32, N, 16> {{
                let shape = AttentionShape::<8, 8>();
                let h = rms_norm(x, b.n1);
                let a = causal_attention(rope(b.wq.dense_forward(h), shape), rope(b.wk.dense_forward(h), shape), b.wv.dense_forward(h), shape);
                let x2 = x + b.wo.dense_forward(a);
                x2 + b.mlp.swiglu_forward(rms_norm(x2, b.n2))
            }}
            fn full(x: [i32; 8], m: G) -> Tensor<f32, 8, 12> {{
                let mut h = embedding_forward(m.tok, x);
                for i in 0..2 {{ h = block(h, m.blocks[i]); }};
                embedding_logits(m.tok, rms_norm(h, m.nf))
            }}
            #[no_inline]
            fn decode_block(x: Tensor<f32, 1, 16>, b: Blk, kc: Tensor<f32, 8, 16>, vc: Tensor<f32, 8, 16>, pos: i32) -> (Tensor<f32, 1, 16>, Tensor<f32, 8, 16>, Tensor<f32, 8, 16>) {{
                let shape = AttentionShape::<8, 8>();
                let h = rms_norm(x, b.n1);
                let k = update(kc, rope_at(b.wk.dense_forward(h), pos, shape), pos, 0);
                let v = update(vc, b.wv.dense_forward(h), pos, 0);
                let a = cached_attention(rope_at(b.wq.dense_forward(h), pos, shape), k, v, pos + 1, shape);
                let x2 = x + b.wo.dense_forward(a);
                (x2 + b.mlp.swiglu_forward(rms_norm(x2, b.n2)), k, v)
            }}
            // The largest gap between the two logits, over the 8 positions.
            fn gap(x: [i32; 8], m: G) -> f32 {{
                let reference = full(x, m);
                let mut keys: [Tensor<f32, 8, 16>; 2] = [for i in 0..2: uninitialized()];
                let mut values: [Tensor<f32, 8, 16>; 2] = [for i in 0..2: uninitialized()];
                let mut worst = 0.0;
                for pos in 0..8 {{
                    let mut h = embedding_forward(m.tok, [x[pos]]);
                    for i in 0..2 {{
                        let (y, k, v) = decode_block(h, m.blocks[i], keys[i], values[i], pos);
                        h = y;
                        keys[i] = k;
                        values[i] = v;
                    }};
                    let z = embedding_logits(m.tok, rms_norm(h, m.nf));
                    for c in 0..12 {{
                        let d = z[0, c] - reference[pos, c];
                        if d > worst {{ worst = d; }};
                        if 0.0 - d > worst {{ worst = 0.0 - d; }};
                    }};
                }};
                worst
            }}
            fn main() -> f32 {{
                rand_seed(3);
                let m = G(tok: Init::xavier(), blocks: [mk(), mk()], nf: ones());
                let x = [3, 1, 4, 1, 5, 9, 2, 6];
                {out}
            }}
        "
        )
    };
    let got = run(&src("gap(x, m)"));
    assert!(got < 1e-4, "decoded logits off by {got}");
}
