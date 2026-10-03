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
