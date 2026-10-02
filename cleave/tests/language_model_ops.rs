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
