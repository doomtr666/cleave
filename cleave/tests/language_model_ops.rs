//! The differentiable operations a character-level language model needs
//! (`doc/plan-nanolm.md`, step 2), each checked by its forward value and its
//! gradient: an `adjoint` rule with a non-differentiable (integer) parameter,
//! embeddings, cross-entropy on integer targets, GELU.

use cleave::cps::{collect_mlir_types, collect_struct_schemas};
use cleave::driver::compile;
use cleave::egraph::{DerivativeRequest, optimize_program, synthesize_derivatives};
use cleave::mlir_lower::lower_program;
use cleave::pipeline::{Backend, CodegenOptions, check_type_errors, lower_to_llvm};
use cleave::refcount::insert_refcounting;
use cleave::registry::Registry;
use melior::Context;
use melior::dialect::DialectRegistry;
use melior::ir::operation::OperationLike;
use melior::utility::register_all_dialects;

/// Compiles `src` through the real pipeline and returns what its `main`
/// returns.
fn run(src: &str) -> f32 {
    let (result, _sources) = compile(vec![("test.cleave".to_string(), src.to_string())], &[]);
    let program = result.unwrap_or_else(|e| panic!("compile failed: {e:?}"));
    let registry = Registry::build(&program);
    if let Err(diags) = check_type_errors(&program, &registry) {
        panic!("type check failed: {diags:?}");
    }
    let units = cleave::cps::collect_units(&program, &registry);
    let requests: Vec<DerivativeRequest> = units
        .iter()
        .filter_map(|u| match &u.body {
            cleave::cps::UnitBody::Derivative(of, is_grad, grad_target_index) => Some(DerivativeRequest {
                name: u.name.clone(),
                of: of.clone(),
                is_grad: *is_grad,
                grad_target_index: *grad_target_index,
            }),
            _ => None,
        })
        .collect();
    let cps_program = cleave::cps::convert_program(units, None);
    let struct_schemas = collect_struct_schemas(&program);
    let cps_program = synthesize_derivatives(cps_program, &requests, &registry, &struct_schemas)
        .unwrap_or_else(|e| panic!("cannot derive: {e:?}"));
    let (cps_program, _) = optimize_program(cps_program, &registry, false);
    let cps_program = cleave::cps::eliminate_dead_code(cps_program);
    let mlir_types = collect_mlir_types(&program);
    let escaping = cleave::escape::escaping_struct_vars(&cps_program);
    let cps_program = insert_refcounting(cps_program, &struct_schemas, &mlir_types, &escaping);

    let dialect_registry = DialectRegistry::new();
    register_all_dialects(&dialect_registry);
    let context = Context::new();
    context.append_dialect_registry(&dialect_registry);
    context.load_all_available_dialects();
    melior::utility::register_all_llvm_translations(&context);
    let mut module = lower_program(&context, &cps_program, &mlir_types, struct_schemas);
    assert!(module.as_operation().verify(), "module failed verification");
    let options = CodegenOptions {
        opt_level: 2,
        openmp: false,
        target_cpu: None,
        target_features: None,
        backend: Backend::Cpu,
        // In-process engine without libomp: spawned calls run in place.
        tasks: false,
        ..Default::default()
    };
    cleave::options::set(options.clone());
    lower_to_llvm(&context, &mut module, &options).expect("lower_to_llvm failed");

    let engine = melior::ExecutionEngine::new(&module, options.opt_level as usize, &[], true, false);
    unsafe {
        use cleave_rt::checkpoint as ck;
        let symbols: &[(&str, *mut ())] = &[
            ("cleave_alloc", cleave_rt::cleave_alloc as *mut ()),
            ("cleave_alloc_rc", cleave_rt::cleave_alloc_rc as *mut ()),
            ("cleave_retain", cleave_rt::cleave_retain as *mut ()),
            ("cleave_release", cleave_rt::cleave_release as *mut ()),
            ("cleave_release_void", cleave_rt::cleave_release_void as *mut ()),
            ("cleave_alloc_local", cleave_rt::cleave_alloc_local as *mut ()),
            ("cleave_region_enter", cleave_rt::cleave_region_enter as *mut ()),
            ("cleave_region_exit", cleave_rt::cleave_region_exit as *mut ()),
            ("cleave_alloc_pool", cleave_rt::cleave_alloc_pool as *mut ()),
            ("cleave_release_pool", cleave_rt::cleave_release_pool as *mut ()),
            ("memrefCopy", cleave_rt::memrefCopy as *mut ()),
            ("cleave_blas_sgemm", cleave_rt::cleave_blas_sgemm as *mut ()),
            ("rand_seed", cleave_rt::rand_seed as *mut ()),
            ("rand_state", cleave_rt::rand_state as *mut ()),
            ("rand_uniform_f32", cleave_rt::rand_uniform_f32 as *mut ()),
            ("rand_normal_f32", cleave_rt::rand_normal_f32 as *mut ()),
            ("cleave_ckpt_create", ck::cleave_ckpt_create as *mut ()),
            ("cleave_ckpt_open", ck::cleave_ckpt_open as *mut ()),
            ("cleave_ckpt_close", ck::cleave_ckpt_close as *mut ()),
            ("cleave_ckpt_write_f32s", ck::cleave_ckpt_write_f32s as *mut ()),
            ("cleave_ckpt_read_f32s", ck::cleave_ckpt_read_f32s as *mut ()),
            ("cleave_ckpt_write_f32", ck::cleave_ckpt_write_f32 as *mut ()),
            ("cleave_ckpt_read_f32", ck::cleave_ckpt_read_f32 as *mut ()),
            ("cleave_ckpt_write_f64", ck::cleave_ckpt_write_f64 as *mut ()),
            ("cleave_ckpt_read_f64", ck::cleave_ckpt_read_f64 as *mut ()),
            ("cleave_ckpt_write_i32", ck::cleave_ckpt_write_i32 as *mut ()),
            ("cleave_ckpt_read_i32", ck::cleave_ckpt_read_i32 as *mut ()),
            ("cleave_ckpt_write_i64", ck::cleave_ckpt_write_i64 as *mut ()),
            ("cleave_ckpt_read_i64", ck::cleave_ckpt_read_i64 as *mut ()),
        ];
        for (name, f) in symbols {
            engine.register_symbol(name, *f);
        }
        let mut result: f32 = 0.0;
        engine
            .invoke_packed("main", &mut [&mut result as *mut f32 as *mut ()])
            .expect("JIT invocation failed");
        result
    }
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
/// (`redundant_copy_elim::lower_dynamic_copies`). Through the CLI because
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
/// loop (`pipeline.rs::has_permuted_transfer`).
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
/// `pipeline.rs`, `cleave_mlir_shim::approximate_math`.
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
/// into it (`redundant_copy_elim.rs`, `forward_dead_source_copies`,
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


/// A light struct of ~57 KB (nested light structs of tensors) passed to a
/// function that isn't inlined. By value, LLVM expanded it into its scalars,
/// stored below the stack pointer with no probe: past Windows' guard page an
/// access violation, depending on the order of the stores and on how much of
/// the stack the thread had committed (nanoLM v2's `Optimizer::step`, 58 KB of
/// arguments; not reproducible on demand in a small program). So what is
/// checked is the ABI itself: the function receives pointers to copies
/// (`mlir_lower.rs::by_pointer`), and still computes the right value.
#[test]
fn a_large_light_struct_crosses_a_call_by_pointer() {
    let src = format!("
        use nn;
        struct L1 {{ t0: Tensor<f32, 1, 1>, t1: Tensor<f32, 1, 1>, t2: Tensor<f32, 1, 1>, t3: Tensor<f32, 1, 1>, t4: Tensor<f32, 1, 1>, t5: Tensor<f32, 1, 1>, t6: Tensor<f32, 1, 1>, t7: Tensor<f32, 1, 1>, t8: Tensor<f32, 1, 1>, t9: Tensor<f32, 1, 1>, t10: Tensor<f32, 1, 1>, t11: Tensor<f32, 1, 1>, t12: Tensor<f32, 1, 1>, t13: Tensor<f32, 1, 1>, t14: Tensor<f32, 1, 1>, t15: Tensor<f32, 1, 1> }}
struct L2 {{ l0: L1, l1: L1, l2: L1, l3: L1, l4: L1, l5: L1, l6: L1, l7: L1, l8: L1, l9: L1, l10: L1, l11: L1, l12: L1, l13: L1, l14: L1, l15: L1 }}
struct L3 {{ a: L2, b: L2, c: L2, d: L2 }}
fn mk1(v: f32) -> L1 {{ L1(t0: [for r in 0..1: [for c in 0..1: v]], t1: [for r in 0..1: [for c in 0..1: v]], t2: [for r in 0..1: [for c in 0..1: v]], t3: [for r in 0..1: [for c in 0..1: v]], t4: [for r in 0..1: [for c in 0..1: v]], t5: [for r in 0..1: [for c in 0..1: v]], t6: [for r in 0..1: [for c in 0..1: v]], t7: [for r in 0..1: [for c in 0..1: v]], t8: [for r in 0..1: [for c in 0..1: v]], t9: [for r in 0..1: [for c in 0..1: v]], t10: [for r in 0..1: [for c in 0..1: v]], t11: [for r in 0..1: [for c in 0..1: v]], t12: [for r in 0..1: [for c in 0..1: v]], t13: [for r in 0..1: [for c in 0..1: v]], t14: [for r in 0..1: [for c in 0..1: v]], t15: [for r in 0..1: [for c in 0..1: v]]) }}
fn mk2(v: f32) -> L2 {{ L2(l0: mk1(v), l1: mk1(v), l2: mk1(v), l3: mk1(v), l4: mk1(v), l5: mk1(v), l6: mk1(v), l7: mk1(v), l8: mk1(v), l9: mk1(v), l10: mk1(v), l11: mk1(v), l12: mk1(v), l13: mk1(v), l14: mk1(v), l15: mk1(v)) }}
        #[no_inline]
        fn read(x: L3, y: L3) -> f32 {{ x.a.l0.t0[0, 0] + y.d.l15.t15[0, 0] }}
        fn main() -> f32 {{
            let x = L3(a: mk2(1.5), b: mk2(0.0), c: mk2(0.0), d: mk2(0.0));
            let y = L3(a: mk2(0.0), b: mk2(0.0), c: mk2(0.0), d: mk2(2.0));
            read(x, y)
        }}
    ");
    let got = run(&src);
    assert!(close(got, 3.5), "got {got}");
    let dir = std::env::temp_dir().join("cleave-language-model-ops");
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join("large_light_struct.cleave");
    std::fs::write(&source, &src).unwrap();
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
