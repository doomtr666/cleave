//! An `extern fn` returning a tensor or an array the host fills: the
//! compiler allocates a fresh buffer and passes it last, as a pointer and a
//! length (`mlir_lower.rs::lower_extern_out_param_call`), so the host side
//! is `fn f(args..., out: *mut T, len: i64)` and the cleave side reads
//! `extern fn f(args...) -> Tensor<T, ...>;` — no buffer to allocate, fill
//! and convert by hand, and no copy.

use cleave::cps::{collect_mlir_types, collect_struct_schemas};
use cleave::driver::compile;
use cleave::egraph::optimize_program;
use cleave::mlir_lower::lower_program;
use cleave::pipeline::{Backend, CodegenOptions, check_type_errors, lower_to_llvm};
use cleave::refcount::insert_refcounting;
use cleave::registry::Registry;
use melior::Context;
use melior::dialect::DialectRegistry;
use melior::ir::operation::OperationLike;
use melior::utility::register_all_dialects;

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

fn run(src: &str) -> i32 {
    let (result, _sources) = compile(vec![("test.cleave".to_string(), src.to_string())], &[]);
    let program = result.unwrap_or_else(|e| panic!("compile failed: {e:?}"));
    let registry = Registry::build(&program);
    if let Err(diags) = check_type_errors(&program, &registry) {
        panic!("type check failed: {diags:?}");
    }
    let units = cleave::cps::collect_units(&program, &registry);
    let cps_program = cleave::cps::convert_program(units, None);
    let (cps_program, _) = optimize_program(cps_program, &registry, false);
    let cps_program = cleave::cps::eliminate_dead_code(cps_program);
    let mlir_types = collect_mlir_types(&program);
    let struct_schemas = collect_struct_schemas(&program);
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
        engine.register_symbol("ramp_f32", ramp_f32 as *mut ());
        engine.register_symbol("ramp_i32", ramp_i32 as *mut ());
        let mut result: i32 = 0;
        engine
            .invoke_packed("main", &mut [&mut result as *mut i32 as *mut ()])
            .expect("JIT invocation failed");
        result
    }
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
