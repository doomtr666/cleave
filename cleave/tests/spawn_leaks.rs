//! Leaks through `spawn` (`doc/plan-spawn.md`): a training loop whose
//! gradient is computed by spawned tasks, measured like `leaks.rs` — the bytes
//! still allocated after N steps must not grow with N. Its own binary, with
//! one test and a process-wide counter: tasks allocate on one thread and free
//! on another, so `leaks.rs`'s per-thread count can't see them, and nothing
//! else may run here. The result of a spawned call used to be left unowned by
//! refcounting, so never released: each step leaked its gradients.
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
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicI64, Ordering};

struct Counting;

static LIVE: AtomicI64 = AtomicI64::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        LIVE.fetch_add(layout.size() as i64, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size() as i64, Ordering::Relaxed);
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        LIVE.fetch_add(new_size as i64 - layout.size() as i64, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// Bytes per step below which a difference is the pool's, not a leak.
const NOISE: i64 = 128;

fn live() -> i64 {
    LIVE.load(Ordering::Relaxed)
}

fn context() -> Context {
    let dialect_registry = DialectRegistry::new();
    register_all_dialects(&dialect_registry);
    let context = Context::new();
    context.append_dialect_registry(&dialect_registry);
    context.load_all_available_dialects();
    context
}

/// One run at a time: the runtime's pool is shared by every thread, so a
/// block one test parks could be handed to another's run, skewing both
/// counts.
static RUN: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Compiles `src` through the real pipeline and runs `main`, returning its
/// result and the bytes the run left allocated.
fn run_counting(src: &str) -> (i32, i64) {
    let _one_at_a_time = RUN.lock().unwrap_or_else(|e| e.into_inner());
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

    let context = context();
    melior::utility::register_all_llvm_translations(&context);
    let mut module = lower_program(&context, &cps_program, &mlir_types, collect_struct_schemas(&program));
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

    // `spawn`'s tasks run on libomp.
    let libomp = format!("{}/bin/libomp.dll", std::env::var("MLIR_SYS_220_PREFIX").expect("MLIR_SYS_220_PREFIX"));
    let engine = cleave_mlir_shim::ExecutionEngine::new(
        module.to_raw(),
        options.opt_level as usize,
        &[libomp.as_str()],
        true,
        false,
        "",
        "",
        true,
    );
    unsafe {
        engine.register_symbol("cleave_alloc", cleave_rt::cleave_alloc as *mut ());
        engine.register_symbol("cleave_alloc_rc", cleave_rt::cleave_alloc_rc as *mut ());
        engine.register_symbol("cleave_retain", cleave_rt::cleave_retain as *mut ());
        engine.register_symbol("cleave_release", cleave_rt::cleave_release as *mut ());
        engine.register_symbol("cleave_release_void", cleave_rt::cleave_release_void as *mut ());
        engine.register_symbol("cleave_alloc_local", cleave_rt::cleave_alloc_local as *mut ());
        engine.register_symbol("cleave_region_enter", cleave_rt::cleave_region_enter as *mut ());
        engine.register_symbol("cleave_region_exit", cleave_rt::cleave_region_exit as *mut ());
        engine.register_symbol("cleave_alloc_pool", cleave_rt::cleave_alloc_pool as *mut ());
        engine.register_symbol("cleave_release_pool", cleave_rt::cleave_release_pool as *mut ());
        engine.register_symbol("memrefCopy", cleave_rt::memrefCopy as *mut ());
        engine.register_symbol("cleave_parallel_threads", cleave_rt::cleave_parallel_threads as *mut ());
        engine.register_symbol("cleave_bind_worker", cleave_rt::cleave_bind_worker as *mut ());
        engine.register_symbol("rand_seed", cleave_rt::rand_seed as *mut ());
        engine.register_symbol("rand_uniform_f32", cleave_rt::rand_uniform_f32 as *mut ());
        engine.register_symbol("rand_normal_f32", cleave_rt::rand_normal_f32 as *mut ());
        engine.register_symbol("cleave_blas_sgemm", cleave_rt::cleave_blas_sgemm as *mut ());
    }
    let before = live();
    let mut result: i32 = 0;
    unsafe {
        engine
            .invoke_packed("main", &mut [&mut result as *mut i32 as *mut ()])
            .expect("JIT invocation failed");
    }
    (result, live() - before)
}

/// A small `mnist-interop`: two `Dense` layers in a `Trainable` struct,
/// `loss` as given, trained `steps` steps.

fn program(steps: u32) -> String {
    format!(
        "
        use nn;
        struct Net {{ l1: Dense<f32, 16, 32>, l2: Dense<f32, 32, 10> }}
        impl Trainable<Net> {{}}
        fn forward(x, net) {{ net.l2.dense_forward(relu(net.l1.dense_forward(x))) }}
        fn loss(x: Tensor<f32, 32, 16>, y: Tensor<f32, 32, 10>, net: Net) -> f32 {{ sum(forward(x, net) - y) }}
        net_grad = grad(loss, net);
        fn par_grad(x: Tensor<f32, 32, 16>, y: Tensor<f32, 32, 10>, net: Net) -> Net {{
            let a = spawn net_grad(x, y, net);
            let b = spawn net_grad(x, y, net);
            accumulate(a, b)
        }}
        fn main() -> i32 {{
            rand_seed(1);
            let mut net = Net(l1: Init::he(), l2: Init::xavier());
            let opt = Sgd(lr: 0.001);
            let mut state = init_state(opt, net);
            let x: Tensor<f32, 32, 16> = Init::he();
            let y: Tensor<f32, 32, 10> = Init::he();
            for s in 0..{steps} {{
                let g = par_grad(x, y, net);
                (net, state) = step(opt, net, g, state);
            }};
            1
        }}
    "
    )
}

#[test]
fn spawned_gradients_dont_leak() {
    // The pool's thread caches hold freed blocks for reuse, a bounded amount,
    // but one that grows over the first steps when blocks are freed on
    // another thread than their allocator's (a task's gradient released by
    // its parent): without them, what's still allocated is exactly what's
    // live. Set before the runtime first reads it.
    unsafe { std::env::set_var("CLEAVE_NO_THREAD_CACHE", "1") };
    let (r_short, short) = run_counting(&program(8));
    let (r_long, long) = run_counting(&program(72));
    assert_eq!((r_short, r_long), (1, 1));
    let per_step = (long - short) / 64;
    assert!(
        per_step < NOISE,
        "{per_step} bytes left allocated per step ({short} after 8 steps, {long} after 72)"
    );
}
