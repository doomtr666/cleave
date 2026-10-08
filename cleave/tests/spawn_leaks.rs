//! Leaks through `spawn` (`doc/plan-spawn.md`): a training loop whose
//! gradient is computed by spawned tasks, measured like `leaks.rs` — the bytes
//! still allocated after N steps must not grow with N. Its own binary, with
//! one test and a process-wide counter: tasks allocate on one thread and free
//! on another, so `leaks.rs`'s per-thread count can't see them, and nothing
//! else may run here. The result of a spawned call used to be left unowned by
//! refcounting, so never released: each step leaked its gradients.
use cleave::pipeline::CodegenOptions;
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

/// One run at a time: the runtime's pool is shared by every thread, so a
/// block one test parks could be handed to another's run, skewing both
/// counts.
static RUN: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Compiles `src` through the real pipeline (`cleave::run`, what `--run`
/// uses, tasks on libomp) and runs `main`, returning its result and the bytes
/// the run left allocated.
fn run_counting(src: &str) -> (i32, i64) {
    let _one_at_a_time = RUN.lock().unwrap_or_else(|e| e.into_inner());
    let options = CodegenOptions { openmp: false, ..Default::default() };
    let (program, registry, sources) = cleave::run::check_sources(vec![("test.cleave".to_string(), src.to_string())], &[], false)
        .unwrap_or_else(|e| panic!("{}", e.join("
")));
    cleave::run::run_main_with(&program, &registry, Some(&sources), &options, &[], |invoke| {
        let before = live();
        let result = invoke().unwrap_or_else(|e| panic!("{}", e.join("
")));
        (result, live() - before)
    })
    .unwrap_or_else(|e| panic!("{}", e.join("
")))
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

/// nanoLM's data-parallel step (`examples/nanolm`'s `parallel_grad`): a model
/// holding an array of layers, its gradient an array of spawned parts summed.
/// Every array built by a function leaked with its elements until 2026-10-08
/// (3.8 GiB a nanoLM step), unseen here: this binary only had the two-part,
/// named-field shape above.
fn array_program(steps: u32) -> String {
    format!(
        "
        use nn;
        struct Layer {{ d: Dense<f32, 16, 16> }}
        impl Trainable<Layer> {{}}
        struct Net {{ layers: [Layer; 2], out: Dense<f32, 16, 16> }}
        impl Trainable<Net> {{}}
        fn new_layer() -> Layer {{ Layer(d: Init::xavier()) }}
        fn forward(x: Tensor<f32, 32, 16>, net: Net) -> Tensor<f32, 32, 16> {{
            let mut h = x;
            for i in 0..2 {{ h = relu(net.layers[i].d.dense_forward(h)); }};
            net.out.dense_forward(h)
        }}
        fn loss(x: Tensor<f32, 32, 16>, y: Tensor<f32, 32, 16>, net: Net) -> f32 {{ sum(forward(x, net) - y) }}
        net_grad = grad(loss, net);
        fn par_grad(x: Tensor<f32, 32, 16>, y: Tensor<f32, 32, 16>, net: Net) -> Net {{
            let parts = [for j in 0..4: spawn net_grad(x, y, net)];
            accumulate(accumulate(parts[0], parts[1]), accumulate(parts[2], parts[3]))
        }}
        fn main() -> i32 {{
            rand_seed(1);
            let a = new_layer();
            let b = new_layer();
            let mut net = Net(layers: [a, b], out: Init::xavier());
            let opt = Sgd(lr: 0.001);
            let mut state = init_state(opt, net);
            let x: Tensor<f32, 32, 16> = Init::he();
            let y: Tensor<f32, 32, 16> = Init::he();
            for s in 0..{steps} {{
                let g = par_grad(x, y, net);
                (net, state) = step(opt, net, g, state);
            }};
            1
        }}
    "
    )
}

/// One test in this binary (its counter is the whole process's): both
/// programs, one after the other.
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

    let (r_short, short) = run_counting(&array_program(8));
    let (r_long, long) = run_counting(&array_program(72));
    assert_eq!((r_short, r_long), (1, 1));
    let per_step = (long - short) / 64;
    assert!(
        per_step < NOISE,
        "an array model: {per_step} bytes left allocated per step ({short} after 8 steps, {long} after 72)"
    );
}
