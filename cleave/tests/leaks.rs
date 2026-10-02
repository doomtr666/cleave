//! Memory leaks in real training loops, measured: a program runs N steps
//! through the real pipeline (`pipeline.rs::lower_to_llvm`, refcounting and
//! the pool allocator included), and the bytes still allocated afterwards
//! must not grow with N. `cleave-rt` allocates through Rust's allocator, so a
//! counting `#[global_allocator]` sees every runtime allocation; counted per
//! thread, so nothing else running in this binary pollutes the figure.
//! The pool keeps freed blocks for reuse, and how many varies by a block or
//! two from run to run, once, not per step: the comparison is between 8 and
//! 72 steps, and anything under `NOISE` bytes per step is that, not a leak
//! (a leaked gradient is thousands of bytes per step).

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
use std::cell::Cell;

struct Counting;

thread_local! {
    static LIVE: Cell<i64> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _ = LIVE.try_with(|l| l.set(l.get() + layout.size() as i64));
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let _ = LIVE.try_with(|l| l.set(l.get() - layout.size() as i64));
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let _ = LIVE.try_with(|l| l.set(l.get() + new_size as i64 - layout.size() as i64));
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// Bytes per step below which a difference is the pool's, not a leak.
const NOISE: i64 = 128;

fn live() -> i64 {
    LIVE.with(|l| l.get())
}

fn context() -> Context {
    let dialect_registry = DialectRegistry::new();
    register_all_dialects(&dialect_registry);
    let context = Context::new();
    context.append_dialect_registry(&dialect_registry);
    context.load_all_available_dialects();
    context
}

/// Compiles `src` through the real pipeline and runs `main`, returning its
/// result and the bytes the run left allocated.
fn run_counting(src: &str) -> (i32, i64) {
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

    let engine = melior::ExecutionEngine::new(&module, options.opt_level as usize, &[], true, false);
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
fn training_program(loss_body: &str, opt: &str, steps: u32) -> String {
    training_program_with(loss_body, opt, steps, "net_grad(x, y, net)")
}

/// `training_program` with the gradient computed by `grad_expr`.
fn training_program_with(loss_body: &str, opt: &str, steps: u32, grad_expr: &str) -> String {
    format!(
        "
        use nn;
        struct Net {{ l1: Dense<f32, 16, 32>, l2: Dense<f32, 32, 10> }}
        impl Trainable<Net> {{}}
        fn forward(x, net) {{ net.l2.dense_forward(relu(net.l1.dense_forward(x))) }}
        fn loss(x: Tensor<f32, 32, 16>, y: Tensor<f32, 32, 10>, net: Net) -> f32 {{ {loss_body} }}
        net_grad = grad(loss, net);
        fn main() -> i32 {{
            rand_seed(1);
            let mut net = Net(l1: Init::he(), l2: Init::xavier());
            let opt = {opt};
            let mut state = init_state(opt, net);
            let x: Tensor<f32, 32, 16> = Init::he();
            let y: Tensor<f32, 32, 10> = Init::he();
            for s in 0..{steps} {{
                let g = {grad_expr};
                (net, state) = step(opt, net, g, state);
            }};
            1
        }}
    "
    )
}

/// Bytes left allocated per training step: the difference between a long
/// and a short run, over the extra steps.
fn leak_per_step(loss_body: &str, opt: &str, grad_expr: &str) -> i64 {
    let (r_short, short) = run_counting(&training_program_with(loss_body, opt, 8, grad_expr));
    let (r_long, long) = run_counting(&training_program_with(loss_body, opt, 72, grad_expr));
    assert_eq!((r_short, r_long), (1, 1));
    (long - short) / 64
}

/// A value holding tensors that leaves an `if` (each branch building its
/// own), in a loop: what the gradient through an `if` reduces to.
fn if_join_program(steps: u32) -> String {
    format!(
        "
        use nn;
        struct Net {{ l1: Dense<f32, 16, 32>, l2: Dense<f32, 32, 10> }}
        fn pick(s: i32) -> f32 {{
            let n = if s >= 0 {{ Net(l1: Init::he(), l2: Init::xavier()) }} else {{ Net(l1: Init::xavier(), l2: Init::he()) }};
            VARIANT
        }}
        fn mk() -> Net {{ Net(l1: Init::he(), l2: Init::xavier()) }}
        fn mix(a: Net, b: Net) -> Net {{ Net(l1: b.l1, l2: a.l2) }}
        fn main() -> i32 {{
            rand_seed(1);
            let mut total = 0.0;
            let mut net = mk();
            for s in 0..{steps} {{ INLOOP }};
            total = total + net.l1.w[0, 0];
            if total == total {{ 1 }} else {{ 0 }}
        }}
    "
    )
}

fn if_join_leak(variant: &str) -> i64 {
    if_join_leak_in(variant, "total = total + pick(s);")
}

fn if_join_leak_in(variant: &str, in_loop: &str) -> i64 {
    let prog = |n| if_join_program(n).replace("VARIANT", variant).replace("INLOOP", in_loop);
    let (a, short) = run_counting(&prog(8));
    let (b, long) = run_counting(&prog(72));
    assert_eq!((a, b), (1, 1));
    (long - short) / 64
}

/// Values holding tensors that leave an `if` (structs of tensors, structs of
/// structs), in a loop body, read afterwards or passed on, or carried by the
/// loop: none of them is left allocated. Found as a whole `Dense` leaked per
/// iteration (`refcount.rs`, `collect_local_free_vars` and the join's own
/// ownership): two `if`s in a row, the first one's value read only inside
/// the second's join.
#[test]
fn values_leaving_an_if_are_released() {
    let _ = run_counting(&if_join_program(1).replace("VARIANT", "1.0").replace("INLOOP", "total = total + pick(s);"));
    let cases = [
        ("a struct of structs, from a fn", "total = total + pick(s);"),
        ("a struct of structs, inline", "let n = if s >= 0 { Net(l1: Init::he(), l2: Init::xavier()) } else { Net(l1: Init::xavier(), l2: Init::he()) }; total = total + n.l2.b[0, 1];"),
        ("two ifs in a row", "let d: Dense<f32, 16, 32> = if s >= 0 { Init::he() } else { Init::xavier() }; let e: Dense<f32, 16, 32> = if s >= 0 { Init::he() } else { Init::xavier() }; total = total + d.w[0, 0] + e.w[0, 0];"),
        ("two ifs, nested struct", "let d: Dense<f32, 16, 32> = if s >= 0 { Init::he() } else { Init::xavier() }; let n = if s >= 0 { Net(l1: Init::he(), l2: Init::xavier()) } else { Net(l1: Init::xavier(), l2: Init::he()) }; total = total + d.w[0, 0] + n.l2.b[0, 1];"),
        ("calls in the branches", "let g = if s >= 0 { mk() } else { mk() }; total = total + g.l1.w[0, 0];"),
        ("passed to a fn", "let g = if s >= 0 { mk() } else { mk() }; let m = mix(net, g); total = total + m.l2.b[0, 0];"),
        ("carried by the loop", "let g = if s >= 0 { mk() } else { mk() }; net = mix(net, g);"),
    ];
    let leaks: Vec<(&str, i64)> = cases
        .iter()
        .map(|(name, body)| (*name, if_join_leak_in("n.l2.b[0, 1]", body)))
        .collect();
    assert!(leaks.iter().all(|(_, l)| *l < NOISE), "bytes leaked per iteration: {leaks:?}");
}

/// A gradient leaving an `if` crashes (`doc/backlog.md`, "A gradient
/// leaving an `if` crashes"): with two call sites, `net_grad` is no longer
/// allocated in its loop's region, and its result through the `if`'s join
/// takes a path the region allocator otherwise always hides.
#[test]
#[ignore = "crashes: doc/backlog.md, A gradient leaving an `if` crashes"]
fn a_gradient_leaving_an_if_does_not_crash() {
    let squares = "let err = forward(x, net) - y; sum(err * err)";
    let through_if = "if s >= 0 { net_grad(x, y, net) } else { net_grad(x, y, net) }";
    let (r, _) = run_counting(&training_program_with(squares, "Sgd(lr: 0.01)", 4, through_if));
    assert_eq!(r, 1);
}

#[test]
fn training_steps_leave_no_allocation_behind() {
    let squares = "let err = forward(x, net) - y; sum(err * err)";
    let entropy = "cross_entropy(forward(x, net), y)";
    let direct = "net_grad(x, y, net)";
    // The runtime's arena is allocated once, on first use: not a leak.
    let _ = run_counting(&training_program(squares, "Sgd(lr: 0.01)", 1));
    let cases = [
        ("squares, Sgd", squares, "Sgd(lr: 0.01)", direct),
        ("cross-entropy, Sgd", entropy, "Sgd(lr: 0.01)", direct),
        ("squares, Momentum", squares, "Momentum(lr: 0.01, beta: 0.9)", direct),
    ];
    let leaks: Vec<(&str, i64)> = cases
        .iter()
        .map(|(name, loss, opt, grad)| (*name, leak_per_step(loss, opt, grad)))
        .collect();
    assert!(leaks.iter().all(|(_, l)| *l < NOISE), "bytes leaked per step: {leaks:?}");
}
