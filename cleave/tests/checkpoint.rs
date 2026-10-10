//! Checkpoints (`stdlib/checkpoint`, `cleave-rt/src/checkpoint.rs`): a value
//! saved and restored is the same value, a run resumed from a checkpoint is
//! the run it would have been, bit for bit, and restoring into a value of
//! another shape is an error that names both shapes.

use cleave::pipeline::CodegenOptions;
use std::path::PathBuf;

/// A path for this test's checkpoint, with forward slashes (a cleave string
/// literal has no escapes for backslashes; Windows takes either).
fn scratch(name: &str) -> String {
    let dir = std::env::temp_dir().join("cleave-checkpoint-tests");
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name).to_string_lossy().replace('\\', "/")
}

/// Compiles and runs `src`'s `main` through the real pipeline
/// (`cleave::run`, what `--run` uses), tasks on: the optimizers spawn.
fn run(src: &str) -> i32 {
    let options = CodegenOptions { openmp: false, tasks: true, ..Default::default() };
    cleave::run::run_source("test.cleave", src, &options).unwrap_or_else(|e| panic!("{}", e.join("
")))
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

/// The same resumed run with a model whose layers are an array
/// (`layers: [Layer; 2]`, `doc/plan-struct-arrays.md`): the array of layers
/// and the optimizer's array of states saved element by element, restored
/// into fresh ones, the run carried on to the same weights to the bit.
#[test]
fn a_run_of_a_model_with_an_array_of_layers_resumes_from_a_checkpoint() {
    let path = scratch("resume_array.ckpt");
    let prelude = "
        use nn;
        use checkpoint;
        struct Layer { d: Dense<f32, 16, 16> }
        impl Trainable<Layer> {}
        struct Net { layers: [Layer; 2], out: Dense<f32, 16, 10> }
        impl Trainable<Net> {}
        fn new_net() -> Net {
            let a = Layer(d: Init::he());
            let b = Layer(d: Init::he());
            Net(layers: [a, b], out: Init::xavier())
        }
        fn forward(x: Tensor<f32, 32, 16>, net: Net) -> Tensor<f32, 32, 10> {
            let mut h = x;
            for i in 0..2 { h = relu(net.layers[i].d.dense_forward(h)); };
            net.out.dense_forward(h)
        }
        fn loss(x: Tensor<f32, 32, 16>, y: Tensor<f32, 32, 10>, net: Net) -> f32 { cross_entropy(forward(x, net), y) }
        net_grad = grad(loss, net);
        fn train(x, y, opt, net, state, steps: i32) {
            let mut n = net;
            let mut s = state;
            for i in 0..steps { (n, s) = step(opt, n, net_grad(x, y, n), s); };
            (n, s)
        }
        fn fingerprint(net: Net) -> f32 {
            sum(net.layers[0].d.w) + sum(net.layers[1].d.w) + sum(net.layers[1].d.b) + sum(net.out.w)
        }
    ";
    let src = format!(
        "{prelude}
        fn main() -> i32 {{
            rand_seed(7);
            let x: Tensor<f32, 32, 16> = Init::he();
            let y: Tensor<f32, 32, 10> = Init::he();
            let opt = Adam(lr: 0.01, beta1: 0.9, beta2: 0.999, eps: 0.00000001);
            let fresh = new_net();

            let (straight, _) = train(x, y, opt, fresh, init_state(opt, fresh), 10);

            rand_seed(7);
            let x2: Tensor<f32, 32, 16> = Init::he();
            let y2: Tensor<f32, 32, 10> = Init::he();
            let first = new_net();
            let (half, half_state) = train(x2, y2, opt, first, init_state(opt, first), 5);
            save(\"{path}\", (half, half_state));

            rand_seed(12345);
            let other = new_net();
            let (net, state) = restore(\"{path}\", (other, init_state(opt, other)));
            let (resumed, _) = train(x2, y2, opt, net, state, 5);

            if fingerprint(resumed) == fingerprint(straight)
                and fingerprint(resumed) != fingerprint(fresh) {{ 1 }} else {{ 0 }}
        }}
    "
    );
    assert_eq!(run_on_a_large_stack(src), 1);
}

/// `run` on a 64 MB stack, as the CLI compiles (`main.rs`): the compiler
/// recurses on the program's continuation-passing form.
fn run_on_a_large_stack(src: String) -> i32 {
    std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(move || run(&src))
        .unwrap()
        .join()
        .unwrap()
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
