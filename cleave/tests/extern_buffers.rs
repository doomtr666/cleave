//! An `extern fn` returning a tensor or an array the host fills: the
//! compiler allocates a fresh buffer and passes it last, as a pointer and a
//! length (`mlir_lower.rs::lower_extern_out_param_call`), so the host side
//! is `fn f(args..., out: *mut T, len: i64)` and the cleave side reads
//! `extern fn f(args...) -> Tensor<T, ...>;` — no buffer to allocate, fill
//! and convert by hand, and no copy.

use cleave::pipeline::CodegenOptions;

/// Writes `start, start + 1, ...` into `out`.
unsafe extern "C" fn ramp_f32(start: i32, out: *mut f32, len: i64) {
    let out = unsafe { std::slice::from_raw_parts_mut(out, len as usize) };
    for (k, x) in out.iter_mut().enumerate() {
        *x = (start + k as i32) as f32;
    }
}

/// The same, as `i32`s.
unsafe extern "C" fn ramp_i32(start: i32, out: *mut i32, len: i64) {
    let out = unsafe { std::slice::from_raw_parts_mut(out, len as usize) };
    for (k, x) in out.iter_mut().enumerate() {
        *x = start + k as i32;
    }
}

/// Compiles and runs `src`'s `main` through the real pipeline
/// (`cleave::run`), the host's `ramp_f32`/`ramp_i32` resolving its externs.
fn run(src: &str) -> i32 {
    let options = CodegenOptions { openmp: false, tasks: false, ..Default::default() };
    let host: &[(&str, *mut ())] = &[("ramp_f32", ramp_f32 as *mut ()), ("ramp_i32", ramp_i32 as *mut ())];
    cleave::run::run_source_with("test.cleave", src, &options, host).unwrap_or_else(|e| panic!("{}", e.join("
")))
}

/// A tensor and an array, each filled by the host; called again in a loop,
/// each call gets its own buffer (the values summed are each call's own).
#[test]
fn an_extern_fn_returns_a_tensor_or_an_array_the_host_fills() {
    let src = "
        use linalg;
        extern fn ramp_f32(start: i32) -> Tensor<f32, 2, 3>;
        extern fn ramp_i32(start: i32) -> [i32; 4];
        fn main() -> i32 {
            let t = ramp_f32(10);
            let a = ramp_i32(5);
            let mut s = 0.0;
            for k in 0..10 {
                let u = ramp_f32(k);
                s = s + u[1, 2];
            };
            if t[0, 0] == 10.0 and t[1, 2] == 15.0 and a[0] == 5 and a[3] == 8 and s == 95.0 { 1 } else { 0 }
        }
    ";
    assert_eq!(run(src), 1);
}
