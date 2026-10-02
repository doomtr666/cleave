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
