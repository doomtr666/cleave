//! `spawn`/`sync` (`doc/plan-spawn.md`): the language side. A spawned call's
//! result is read like any value; every execution must give the result of the
//! serial program (serial elision), whatever runs in parallel.

fn run(name: &str, src: &str) -> String {
    let dir = std::env::temp_dir().join("cleave-spawn");
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join(format!("{name}.cleave"));
    std::fs::write(&source, src).unwrap();
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_cleave"))
        .args(["--no-openmp", "--no-debug-info", "--run"])
        .arg(&source)
        .output()
        .expect("cannot run cleave");
    // The process exits with `main`'s value: read the printed result instead.
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stdout.contains("main returned:"), "stdout: {stdout}\nstderr: {stderr}");
    stdout
}

#[test]
fn spawned_calls_give_their_results() {
    let out = run(
        "two_spawns",
        "
        fn work(n: i32) -> i32 {
            let mut s = 0;
            for i in 0..n { s = s + i; };
            s
        }
        fn main() -> i32 {
            let a = spawn work(10);
            let b = spawn work(20);
            a + b
        }
        ",
    );
    // 45 + 190
    assert!(out.contains("main returned: 235"), "{out}");
}

#[test]
fn spawn_resolves_generic_functions_and_algebra_methods() {
    let out = run(
        "generic_spawn",
        "
        use nn;
        fn twice<T: Ring>(x: T) -> T { x + x }
        fn main() -> i32 {
            let a = spawn twice(21);
            let t: Tensor<f32, 4, 4> = [for i in 0..4: [for j in 0..4: 1.5]];
            let u = spawn Ring::add(t, t);
            sync;
            if u[2, 3] == 3.0 { a } else { 0 }
        }
        ",
    );
    assert!(out.contains("main returned: 42"), "{out}");
}

#[test]
fn a_function_named_spawn_is_still_an_ordinary_call() {
    let out = run(
        "named_spawn",
        "
        fn spawn(x: i32) -> i32 { x * 2 }
        fn main() -> i32 { spawn(21) }
        ",
    );
    assert!(out.contains("main returned: 42"), "{out}");
}

/// Spawned calls become OpenMP tasks after bufferization
/// (`cleave_mlir_shim::lower_spawns`): each in an `omp.task`, each wait an
/// `omp.taskwait`, and the spawning function wrapped to open a parallel region
/// when not already in one.
#[test]
fn spawned_calls_become_openmp_tasks() {
    let dir = std::env::temp_dir().join("cleave-spawn");
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join("tasks_ir.cleave");
    let dump = dir.join("tasks_ir_post_dealloc.mlir");
    let _ = std::fs::remove_file(&dump);
    std::fs::write(
        &source,
        "
        fn work(n: i32) -> i32 { n * 3 }
        fn main() -> i32 {
            let a = spawn work(10);
            let b = spawn work(4);
            a + b
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
    assert!(stdout.contains("main returned: 42"), "stdout: {stdout}\nstderr: {}", String::from_utf8_lossy(&output.stderr));
    let ir = std::fs::read_to_string(&dump).expect("no IR dumped");
    assert_eq!(ir.matches("omp.task {").count(), 2, "{ir}");
    assert!(ir.contains("omp.taskwait"), "{ir}");
    assert!(ir.contains("omp.parallel") && ir.contains("omp.single") && ir.contains("main$tasks"), "{ir}");
    assert!(!ir.contains("cleave_spawn_next") && !ir.contains("cleave_task_wait"), "markers left: {ir}");
}

/// Tensors through tasks: a spawned call's argument built just before the
/// spawn and never read again must stay alive until the task has run (the
/// wait markers keep it), and tensor results come back intact.
#[test]
fn tensors_go_through_spawned_calls() {
    let out = run(
        "tensor_spawns",
        "
        use nn;
        fn scaled(t: Tensor<f32, 64, 64>, s: f32) -> Tensor<f32, 64, 64> {
            let mut acc = t;
            for i in 0..20 { acc = acc + Scale::scale(t, s); };
            acc
        }
        fn main() -> i32 {
            let a = spawn scaled([for i in 0..64: [for j in 0..64: 1.0]], 0.5);
            let b = spawn scaled([for i in 0..64: [for j in 0..64: 2.0]], 0.25);
            let c = a + b;
            if c[3, 4] == 23.0 and c[63, 63] == 23.0 { 0 } else { 1 }
        }
        ",
    );
    // a = 1 + 20 * (1 * 0.5) = 11, b = 2 + 20 * (2 * 0.25) = 12
    assert!(out.contains("main returned: 0"), "{out}");
}

/// A spawned function that spawns: its tasks run on the team the outermost
/// spawning function opened.
#[test]
fn spawned_functions_can_spawn() {
    let out = run(
        "nested_spawns",
        "
        fn leaf(n: i32) -> i32 { n + 1 }
        fn pair(n: i32) -> i32 {
            let a = spawn leaf(n);
            let b = spawn leaf(n * 10);
            a + b
        }
        fn main() -> i32 {
            let x = spawn pair(1);
            let y = spawn pair(2);
            x + y
        }
        ",
    );
    // pair(1) = 2 + 11 = 13, pair(2) = 3 + 21 = 24
    assert!(out.contains("main returned: 37"), "{out}");
}

/// This first version waits for every task at the function's end, which
/// needs every spawned value in scope there: a `spawn` inside a branch, a
/// loop, a block or another expression is a located error, not a crash.
#[test]
fn a_spawn_outside_a_top_level_let_is_a_located_error() {
    let dir = std::env::temp_dir().join("cleave-spawn");
    std::fs::create_dir_all(&dir).unwrap();
    for (name, body) in [
        ("in_branch", "if n > 0 { let a = spawn work(n); a } else { 0 }"),
        ("in_loop", "let mut s = 0; for i in 0..3 { let a = spawn work(i); s = s + a; }; s"),
        ("in_expression", "work(1) + spawn work(2)"),
        ("in_arguments", "let a = spawn work(spawn work(1)); a"),
        // A run-time bound: one task per element needs the elements unrolled.
        ("in_runtime_comprehension", "let xs = [for i in 0..n: spawn work(i)]; xs[0]"),
    ] {
        let source = dir.join(format!("misplaced_{name}.cleave"));
        std::fs::write(
            &source,
            format!("fn work(n: i32) -> i32 {{ n * 2 }}\nfn f(n: i32) -> i32 {{ {body} }}\nfn main() -> i32 {{ f(3) }}\n"),
        )
        .unwrap();
        let output = std::process::Command::new(env!("CARGO_BIN_EXE_cleave"))
            .args(["--no-openmp", "--no-debug-info", "--run"])
            .arg(&source)
            .output()
            .expect("cannot run cleave");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains("`spawn` must be the value of a `let`") && stderr.contains(":2:"),
            "{name}: expected a located MisplacedSpawn error, got:\n{stderr}"
        );
    }
}

/// A spawned call reaching an impure extern (here the random generator, one
/// global stream) is an error naming the path (`cps::check_spawn_purity`): a
/// task runs in parallel and must not touch global state.
#[test]
fn a_spawned_call_reaching_global_state_is_an_error() {
    let dir = std::env::temp_dir().join("cleave-spawn");
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join("impure_spawn.cleave");
    std::fs::write(
        &source,
        "
        use nn;
        fn noise() -> Tensor<f32, 8, 8> { Init::he() }
        fn main() -> i32 {
            let a = spawn noise();
            if a[0, 0] == 12345.0 { 1 } else { 0 }
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
    assert!(
        stderr.contains("`main` spawns `noise`") && stderr.contains("neither `#[pure]` nor `#[reentrant]`") && stderr.contains("rand_normal"),
        "expected the purity error, got:\n{stderr}"
    );
}

/// `[for i in 0..m.len(): spawn f(m[i])]` over a `Trainable` model: one task
/// per field, nested models recursively (`par_sum` sums two models this way).
#[test]
fn a_comprehension_of_spawns_runs_one_task_per_field() {
    let out = run(
        "field_spawns",
        "
        use nn;
        struct Net { a: Tensor<f32, 2, 3>, d: Dense<f32, 3, 2> }
        impl Trainable<Net> {}
        fn mk(v: f32) -> Net {
            Net(a: [for i in 0..2: [for j in 0..3: v]], d: Dense(w: [for i in 0..3: [for j in 0..2: v * 2.0]], b: [for i in 0..1: [for j in 0..2: v * 3.0]]))
        }
        algebra ParSum<M> { fn par_sum(a: M, b: M) -> M; }
        impl<T: Float + Ring, const Dims...: i32> ParSum<Tensor<T, Dims...>> { fn par_sum(a, b) { a + b } }
        impl<M: Trainable> ParSum<M> {
            fn par_sum(a, b) {
                let sums = [for i in 0..a.len(): spawn par_sum(a[i], b[i])];
                sums
            }
        }
        fn main() -> i32 {
            let s = par_sum(mk(1.0), mk(2.0));
            if s.a[1, 2] == 3.0 and s.d.w[2, 1] == 6.0 and s.d.b[0, 1] == 9.0 { 0 } else { 1 }
        }
        ",
    );
    assert!(out.contains("main returned: 0"), "{out}");
}

/// Every task holds the spawned call, and only it: the storage a result goes
/// into (a tuple element's `cleave_alloc_rc`) is allocated right before the
/// call, between it and its marker, and was once taken for the spawned call —
/// the allocation ran as a task, read back before it had run.
#[test]
fn each_task_holds_its_spawned_call() {
    let dir = std::env::temp_dir().join("cleave-spawn");
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join("field_spawns_ir.cleave");
    let dump = dir.join("field_spawns_ir_post_dealloc.mlir");
    let _ = std::fs::remove_file(&dump);
    std::fs::write(
        &source,
        "
        use nn;
        struct Net { a: Tensor<f32, 2, 3>, d: Dense<f32, 3, 2> }
        impl Trainable<Net> {}
        fn mk(v: f32) -> Net {
            Net(a: [for i in 0..2: [for j in 0..3: v]], d: Dense(w: [for i in 0..3: [for j in 0..2: v * 2.0]], b: [for i in 0..1: [for j in 0..2: v * 3.0]]))
        }
        algebra ParSum<M> { fn par_sum(a: M, b: M) -> M; }
        impl<T: Float + Ring, const Dims...: i32> ParSum<Tensor<T, Dims...>> { fn par_sum(a, b) { a + b } }
        impl<M: Trainable> ParSum<M> {
            fn par_sum(a, b) {
                let sums = [for i in 0..a.len(): spawn par_sum(a[i], b[i])];
                sums
            }
        }
        fn main() -> i32 {
            let s = par_sum(mk(1.0), mk(2.0));
            if s.a[1, 2] == 3.0 { 0 } else { 1 }
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
    assert!(String::from_utf8_lossy(&output.stdout).contains("main returned: 0"));
    let ir = std::fs::read_to_string(&dump).expect("no IR dumped");
    // The calls inside each `omp.task { ... }` region, by indentation.
    let mut tasks: Vec<Vec<String>> = Vec::new();
    let mut open: Option<usize> = None;
    for line in ir.lines() {
        let indent = line.len() - line.trim_start().len();
        match open {
            None if line.trim_start().starts_with("omp.task {") => {
                open = Some(indent);
                tasks.push(Vec::new());
            }
            Some(at) if indent == at && line.trim() == "}" => open = None,
            Some(_) if line.contains("call @") => tasks.last_mut().unwrap().push(line.trim().to_string()),
            _ => {}
        }
    }
    assert_eq!(tasks.len(), 4, "one task per field of `Net` and of its `Dense`:\n{ir}");
    for calls in &tasks {
        assert!(
            calls.len() == 1 && calls[0].contains("ParSum::par_sum"),
            "a task holding something else than its spawned call: {calls:?}"
        );
    }
}

/// `[for j in 0..N: spawn f(..)]` over a numeric range with a `define`d bound,
/// arguments computed from the index (nanoLM's parallel evaluation): one task
/// per element, the array read like any other once built.
#[test]
fn a_comprehension_of_spawns_over_a_numeric_range() {
    let out = run(
        "range_comprehension",
        "
        use convert;
        define PARTS: i32 = 4;
        fn work(n: i32) -> f32 {
            let mut s = 0.0;
            for i in 0..n { s = s + 1.0; };
            s
        }
        fn part(base: i32, j: i32) -> i32 { base * (j + 1) }
        fn total(base: i32) -> f64 {
            let parts = [for j in 0..PARTS: spawn work(part(base, j))];
            let mut t: f64 = 0.0;
            for j in 0..PARTS { t = t + parts[j].to(); };
            t
        }
        fn main() -> i32 {
            let t = total(10);
            if t == 100.0 { 1 } else { 0 }
        }
        ",
    );
    // 10 + 20 + 30 + 40
    assert!(out.contains("main returned: 1"), "{out}");
}
