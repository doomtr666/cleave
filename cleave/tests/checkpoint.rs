//! Checkpoints (`stdlib/checkpoint`, `cleave-rt/src/checkpoint.rs`): a value
//! saved and restored is the same value, a run resumed from a checkpoint is
//! the run it would have been, bit for bit, and restoring into a value of
//! another shape is an error that names both shapes.

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
use std::path::PathBuf;

/// A path for this test's checkpoint, with forward slashes (a cleave string
/// literal has no escapes for backslashes; Windows takes either).
fn scratch(name: &str) -> String {
    let dir = std::env::temp_dir().join("cleave-checkpoint-tests");
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name).to_string_lossy().replace('\\', "/")
}

fn run(src: &str) -> i32 {
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
        let mut result: i32 = 0;
        engine
            .invoke_packed("main", &mut [&mut result as *mut i32 as *mut ()])
            .expect("JIT invocation failed");
        result
    }
}

/// Scalars, tensors of rank 1 and 2, in a tuple: restored as saved, into a
/// value of the same shape holding something else.
#[test]
fn a_saved_value_is_restored_as_it_was() {
    let path = scratch("roundtrip.ckpt");
    let src = format!(
        "
        use checkpoint;
        fn main() -> i32 {{
            let v = Tensor::<f32, 3>(data: [1.5, -2.0, 3.25]);
            let m = Tensor::<f32, 2, 2>(data: [[1.0, 2.0], [3.0, 4.0]]);
            save(\"{path}\", (0.5, 7, v, m, 9:i64));
            let zero3 = Tensor::<f32, 3>(data: [0.0, 0.0, 0.0]);
            let zero22 = Tensor::<f32, 2, 2>(data: [[0.0, 0.0], [0.0, 0.0]]);
            let (a, b, c, d, e) = restore(\"{path}\", (0.0, 0, zero3, zero22, 0:i64));
            if a == 0.5 and b == 7 and c[2] == 3.25 and d[1, 0] == 3.0 and e == 9:i64 {{ 1 }} else {{ 0 }}
        }}
    "
    );
    assert_eq!(run(&src), 1);
}

/// A small `Trainable` model trained with Adam: 10 steps straight, and 5
/// steps, a checkpoint (model, optimizer state, random generator), a restore
/// into a freshly initialized model and 5 more steps, end with the same
/// weights to the bit and the same next random number.
#[test]
fn a_run_resumed_from_a_checkpoint_is_the_same_run() {
    let path = scratch("resume.ckpt");
    let prelude = "
        use nn;
        use checkpoint;
        struct Net { l1: Dense<f32, 16, 32>, l2: Dense<f32, 32, 10> }
        impl Trainable<Net> {}
        fn forward(x, net) { net.l2.dense_forward(relu(net.l1.dense_forward(x))) }
        fn loss(x: Tensor<f32, 32, 16>, y: Tensor<f32, 32, 10>, net: Net) -> f32 { cross_entropy(forward(x, net), y) }
        net_grad = grad(loss, net);
        fn train(x, y, opt, net, state, steps: i32) {
            let mut n = net;
            let mut s = state;
            for i in 0..steps { (n, s) = step(opt, n, net_grad(x, y, n), s); };
            (n, s)
        }
        fn fingerprint(net: Net) -> f32 { sum(net.l1.w) + sum(net.l2.w) + sum(net.l2.b) }
    ";
    let src = format!(
        "{prelude}
        fn main() -> i32 {{
            rand_seed(7);
            let x: Tensor<f32, 32, 16> = Init::he();
            let y: Tensor<f32, 32, 10> = Init::he();
            let opt = Adam(lr: 0.01, beta1: 0.9, beta2: 0.999, eps: 0.00000001);
            let fresh = Net(l1: Init::he(), l2: Init::xavier());

            let (straight, _) = train(x, y, opt, fresh, init_state(opt, fresh), 10);
            let straight_next = rand_uniform_f32();

            rand_seed(7);
            let x2: Tensor<f32, 32, 16> = Init::he();
            let y2: Tensor<f32, 32, 10> = Init::he();
            let first = Net(l1: Init::he(), l2: Init::xavier());
            let (half, half_state) = train(x2, y2, opt, first, init_state(opt, first), 5);
            save(\"{path}\", (half, half_state, rand_state()));

            rand_seed(12345);
            let other = Net(l1: Init::he(), l2: Init::xavier());
            let (net, state, seed) = restore(\"{path}\", (other, init_state(opt, other), 0:i64));
            rand_seed(seed);
            let (resumed, _) = train(x2, y2, opt, net, state, 5);
            let resumed_next = rand_uniform_f32();

            if fingerprint(resumed) == fingerprint(straight) and resumed_next == straight_next
                and fingerprint(resumed) != fingerprint(fresh) {{ 1 }} else {{ 0 }}
        }}
    "
    );
    assert_eq!(run(&src), 1);
}

/// Restoring into a value of another shape stops the program with a message
/// naming the shape found and the shape expected (run through the real CLI:
/// the error ends the process).
#[test]
fn restoring_into_another_shape_is_a_clear_error() {
    let path = scratch("mismatch.ckpt");
    let dir = std::env::temp_dir().join("cleave-checkpoint-tests");
    let source: PathBuf = dir.join("mismatch.cleave");
    std::fs::write(
        &source,
        format!(
            "
            use checkpoint;
            fn main() -> i32 {{
                save(\"{path}\", Tensor::<f32, 2, 3>(data: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]));
                let t = restore(\"{path}\", Tensor::<f32, 3, 2>(data: [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]));
                1
            }}
        "
        ),
    )
    .unwrap();
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_cleave"))
        .args(["--no-openmp", "--run"])
        .arg(&source)
        .output()
        .expect("cannot run cleave");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success(), "a mismatched restore must fail");
    assert!(
        stderr.contains("f32[2, 3]") && stderr.contains("expects f32[3, 2]"),
        "got: {stderr}"
    );
}
