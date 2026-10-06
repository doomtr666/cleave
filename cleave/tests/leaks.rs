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
        // In-process engine without libomp: spawned calls run in place.
        tasks: false,
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

/// A loop carrying a tensor whose new value is a fresh buffer from `sgemm`
/// (`x = blas_matmul(w, x)`): One-Shot Bufferize used to reject it outright
/// (`allow-return-allocs-from-loops`); now each iteration's buffer replaces the
/// last, and the one left behind must be freed, not accumulated.
#[test]
fn a_loop_carrying_a_blas_result_frees_each_iterations_buffer() {
    let program = |steps: u32| {
        format!(
            "
            use nn;
            fn main() -> i32 {{
                let o: f32 = 0.001;
                let w: Tensor<f32, 384, 384> = mlir::tensor::splat(o);
                let mut x: Tensor<f32, 384, 384> = mlir::tensor::splat(o);
                for i in 0..{steps} {{ x = blas_matmul(w, x); }};
                if x[0, 0] >= 0.0 {{ 1 }} else {{ 0 }}
            }}
            "
        )
    };
    let (r8, live8) = run_counting(&program(8));
    let (r72, live72) = run_counting(&program(72));
    assert_eq!((r8, r72), (1, 1));
    let per_iteration = (live72 - live8) / 64;
    assert!(per_iteration < NOISE, "{per_iteration} bytes left behind per iteration");
}

/// One loop of `steps` iterations of `body` on `m` (a `Tensor<f32, 64, 48>`
/// and its gradient `g`), counting what's left allocated: per iteration.
fn leak_per_iteration(prelude: &str, body: &str) -> i64 {
    let program = |steps: u32| {
        format!(
            "
            use nn;
            {prelude}
            fn main() -> i32 {{
                rand_seed(1);
                let mut m: Tensor<f32, 64, 48> = Init::xavier();
                let g: Tensor<f32, 64, 48> = Init::xavier();
                {body}
                if m[0, 0] == m[0, 0] {{ 1 }} else {{ 0 }}
            }}
            "
        )
        .replace("STEPS", &steps.to_string())
    };
    let (r8, live8) = run_counting(&program(8));
    let (r72, live72) = run_counting(&program(72));
    assert_eq!((r8, r72), (1, 1));
    (live72 - live8) / 64
}

#[test]
fn muon_steps_leave_no_allocation_behind() {
    let per = leak_per_iteration(
        "",
        "let opt = Muon(lr: 0.01, momentum: 0.95, weight_decay: 0.0,
                       adamw: AdamW(lr: 0.001, beta1: 0.9, beta2: 0.95, eps: 0.00000001, weight_decay: 0.0));
         let mut s = init_state(opt, m);
         for i in 0..STEPS { (m, s) = step(opt, m, g, s); };",
    );
    assert!(per < NOISE, "{per} bytes per Muon step");
}

#[test]
fn clipping_leaves_no_allocation_behind() {
    let per = leak_per_iteration(
        "struct Net { d: Dense<f32, 64, 48> }\n impl Trainable<Net> {}",
        "let mut n = Net(d: Dense(w: m, b: Init::xavier()));
         for i in 0..STEPS { n = clip_grad_norm(n, 0.5); };
         m = n.d.w;",
    );
    assert!(per < NOISE, "{per} bytes per clip");
}

/// nanoLM's shape: each step clips a fresh gradient, then consumes it. Early
/// in training every gradient is scaled; later its norm falls under the
/// threshold and the `else` branch returns it as is (the night run's leak,
/// 18 GB at step 300, 76 GB at step 9100).
fn clipping_a_fresh_gradient(max_norm: &str) -> i64 {
    leak_per_iteration(
        "struct Net { d: Dense<f32, 64, 48> }
 impl Trainable<Net> {}",
        &format!(
            "let b: Tensor<f32, 1, 48> = Init::xavier();
             for i in 0..STEPS {{
                 let c = clip_grad_norm(Net(d: Dense(w: g, b: b)), {max_norm});
                 m = m + c.d.w;
             }};"
        ),
    )
}

/// The generic shape behind `clip_grad_norm`'s leak: a function returning
/// its (borrowed) parameter on one path and a fresh value on the other.
/// Its callers took the result for the argument itself and kept from
/// releasing it, so the fresh path leaked the argument; the parameter is
/// now retained where it's returned (`refcount.rs::returned_param_args`).
#[test]
fn returning_a_parameter_on_one_path_only_leaves_nothing_behind() {
    let per = leak_per_iteration(
        "struct Net { d: Dense<f32, 64, 48> }
         fn rebuilt_or_kept(g: Net, t: f32) -> Net {
             if t < 1000.0 { Net(d: Dense(w: g.d.w + g.d.w, b: g.d.b)) } else { g }
         }",
        "let b: Tensor<f32, 1, 48> = Init::xavier();
         for i in 0..STEPS {
             let rebuilt = rebuilt_or_kept(Net(d: Dense(w: g, b: b)), 0.5);
             let kept = rebuilt_or_kept(Net(d: Dense(w: g, b: b)), 5000.0);
             m = m + rebuilt.d.w + kept.d.w;
         };",
    );
    assert!(per < NOISE, "{per} bytes per pair of calls");
}

#[test]
fn clipping_a_fresh_gradient_that_is_scaled_leaves_no_allocation_behind() {
    let per = clipping_a_fresh_gradient("0.001");
    assert!(per < NOISE, "{per} bytes per clip");
}

#[test]
fn clipping_a_fresh_gradient_that_is_kept_leaves_no_allocation_behind() {
    let per = clipping_a_fresh_gradient("1000000.0");
    assert!(per < NOISE, "{per} bytes per clip");
}

#[test]
fn a_tied_embedding_gradient_leaves_no_allocation_behind() {
    let per = leak_per_iteration(
        "fn loss(e: Embedding<f32, 64, 48>) -> f32 {
             let ids: [i32; 8] = [1, 5, 9, 2, 63, 0, 7, 7];
             sum(embedding_logits(e, embedding_forward(e, ids)))
         }
         de = grad(loss, e);",
        "let mut e = Embedding(table: m);
         for i in 0..STEPS { let d = de(e); e = Embedding(table: e.table - Scale::scale(d.table, 0.0001)); };
         m = e.table;",
    );
    assert!(per < NOISE, "{per} bytes per gradient");
}

#[test]
fn transposes_leave_no_allocation_behind() {
    let per = leak_per_iteration("", "for i in 0..STEPS { m = transpose(transpose(m)); };");
    assert!(per < NOISE, "{per} bytes per pair of transposes");
}

#[test]
fn newton_schulz_leaves_no_allocation_behind() {
    let per = leak_per_iteration("", "for i in 0..STEPS { m = newton_schulz5(m); };");
    assert!(per < NOISE, "{per} bytes per orthogonalization");
}

const CARRIED: &str = "
    fn next_tensor<const R: i32, const C: i32>(m: Tensor<f32, R, C>, g: Tensor<f32, R, C>) -> Tensor<f32, R, C> {
        m - Scale::scale(g, 0.001)
    }
    fn next_state<const R: i32, const C: i32>(g: Tensor<f32, R, C>, s: AdamState<f32, R, C>) -> AdamState<f32, R, C> {
        AdamState::<f32, R, C>(m: Scale::scale(s.m, 0.9) + g, v: s.v, beta1_pow: s.beta1_pow, beta2_pow: s.beta2_pow)
    }
    fn next_both<const R: i32, const C: i32>(m: Tensor<f32, R, C>, g: Tensor<f32, R, C>, s: AdamState<f32, R, C>) -> (Tensor<f32, R, C>, AdamState<f32, R, C>) {
        (next_tensor(m, g), next_state(g, s))
    }
";

#[test]
fn carried_1_a_tensor_from_a_call() {
    let per = leak_per_iteration(CARRIED, "for i in 0..STEPS { m = next_tensor(m, g); };");
    assert!(per < NOISE, "{per} bytes per step");
}

#[test]
fn carried_2_a_struct_rebuilt() {
    let per = leak_per_iteration(
        CARRIED,
        "let mut s = AdamState::<f32, 64, 48>(m: Ring::zero(), v: Ring::zero(), beta1_pow: 1.0, beta2_pow: 1.0);
         for i in 0..STEPS { s = next_state(g, s); };
         m = s.m;",
    );
    assert!(per < NOISE, "{per} bytes per step");
}

#[test]
fn carried_3_both_from_a_tuple() {
    let per = leak_per_iteration(
        CARRIED,
        "let mut s = AdamState::<f32, 64, 48>(m: Ring::zero(), v: Ring::zero(), beta1_pow: 1.0, beta2_pow: 1.0);
         for i in 0..STEPS { (m, s) = next_both(m, g, s); };",
    );
    assert!(per < NOISE, "{per} bytes per step");
}

fn model_steps_leak(opt: &str) -> i64 {
    leak_per_iteration(
        "struct Net { e: Embedding<f32, 64, 48>, d: Dense<f32, 48, 64> }\n impl Trainable<Net> {}",
        &format!(
            "let opt = {opt};
             let mut net = Net(e: Embedding(table: m), d: Init::xavier());
             let grad = Net(e: Embedding(table: g), d: Init::xavier());
             let mut st = init_state(opt, net);
             for i in 0..STEPS {{ (net, st) = step(opt, net, grad, st); }};
             m = net.e.table;"
        ),
    )
}

#[test]
fn model_steps_under_adamw_leave_nothing_behind() {
    let per = model_steps_leak("AdamW(lr: 0.001, beta1: 0.9, beta2: 0.95, eps: 0.00000001, weight_decay: 0.0)");
    assert!(per < NOISE, "{per} bytes per step");
}

#[test]
fn model_steps_under_muon_leave_nothing_behind() {
    let per = model_steps_leak(
        "Muon(lr: 0.01, momentum: 0.95, weight_decay: 0.0,
              adamw: AdamW(lr: 0.001, beta1: 0.9, beta2: 0.95, eps: 0.00000001, weight_decay: 0.0))",
    );
    assert!(per < NOISE, "{per} bytes per step");
}

#[test]
fn accumulating_embeddings_leaves_nothing_behind() {
    let per = leak_per_iteration(
        "",
        "let mut e = Embedding(table: m);
         let d = Embedding(table: g);
         for i in 0..STEPS { e = Embedding(table: Scale::scale(accumulate(e, d).table, 0.5)); };
         m = e.table;",
    );
    assert!(per < NOISE, "{per} bytes per step");
}

#[test]
fn a_modern_block_gradient_leaves_nothing_behind() {
    let per = leak_per_iteration(
        "struct B { n: Tensor<f32, 1, 48>, wq: Dense<f32, 48, 48>, mlp: SwiGlu<f32, 48, 32> }
         impl Trainable<B> {}
         fn ones() -> Tensor<f32, 1, 48> { [for i in 0..1: [for j in 0..48: 1.0]] }
         fn loss(x: Tensor<f32, 64, 48>, b: B) -> f32 {
             let shape = AttentionShape::<16, 16>();
             let h = rms_norm(x, b.n);
             let q = rope(b.wq.dense_forward(h), shape);
             sum(b.mlp.swiglu_forward(q))
         }
         db = grad(loss, b);",
        "let mut b = B(n: ones(), wq: Init::xavier(), mlp: Init::xavier());
         for i in 0..STEPS { let d = db(m, b); b = B(n: b.n, wq: b.wq, mlp: SwiGlu(gate: b.mlp.gate, up: b.mlp.up, down: Dense(w: b.mlp.down.w - Scale::scale(d.mlp.down.w, 0.0001), b: b.mlp.down.b))); };",
    );
    assert!(per < NOISE, "{per} bytes per step");
}

/// `[for j in 0..N: spawn f(..)]` collecting structs, read by index
/// (nanoLM's `parallel_grad`: eight micro-batch gradients, then summed).
#[test]
fn a_comprehension_of_spawned_structs_leaves_nothing_behind() {
    let per = leak_per_iteration(
        "struct Net { d: Dense<f32, 64, 48> }
         impl Trainable<Net> {}
         fn part(m: Tensor<f32, 64, 48>, j: i32) -> Net {
             let s: f32 = j.to();
             Net(d: Dense(w: Scale::scale(m, s), b: Ring::zero()))
         }
         fn total(m: Tensor<f32, 64, 48>) -> Net {
             let g = [for j in 0..4: spawn part(m, j)];
             accumulate(accumulate(g[0], g[1]), accumulate(g[2], g[3]))
         }",
        "for i in 0..STEPS { m = Scale::scale(total(m).d.w, 0.1); };",
    );
    assert!(per < NOISE, "{per} bytes per step");
}

/// The same as `a_comprehension_of_spawned_structs_leaves_nothing_behind`,
/// sequential: a tuple built from locally owned structs, read by index.
#[test]
fn a_tuple_of_owned_structs_leaves_nothing_behind() {
    let per = leak_per_iteration(
        "struct Net { d: Dense<f32, 64, 48> }
         impl Trainable<Net> {}
         fn part(m: Tensor<f32, 64, 48>, j: i32) -> Net {
             let s: f32 = j.to();
             Net(d: Dense(w: Scale::scale(m, s), b: Ring::zero()))
         }
         fn total(m: Tensor<f32, 64, 48>) -> Net {
             let a = part(m, 0);
             let b = part(m, 1);
             let g = (a, b);
             accumulate(g[0], g[1])
         }",
        "for i in 0..STEPS { m = Scale::scale(total(m).d.w, 0.1); };",
    );
    assert!(per < NOISE, "{per} bytes per step");
}

