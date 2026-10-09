//! JIT-executed tests for `cleave-unify-tensor-allocations` (`cleave-mlir-
//! shim`): the `malloc`/`free` calls bufferization leaves made calls to
//! cleave's allocator. A self-contained harness replicating `pipeline.rs`'s
//! stage order through to the `llvm` dialect, since the pass runs after
//! `--convert-to-llvm`, where those calls first exist as such.

use cleave::cps::{collect_mlir_types, collect_struct_schemas, collect_units, convert_program};
use cleave::driver::compile;
use cleave::mlir_lower::lower_program;
use cleave::pipeline::{check_type_errors, strip_ciface_wrapper_debug_info};
use cleave::registry::Registry;
use cleave_mlir_shim::mlir::Context;
use cleave_mlir_shim::mlir::dialect::DialectRegistry;
use cleave_mlir_shim::mlir::utility::register_all_dialects;

fn context() -> Context {
    let dialect_registry = DialectRegistry::new();
    register_all_dialects(&dialect_registry);
    let context = Context::new();
    context.append_dialect_registry(&dialect_registry);
    context.load_all_available_dialects();
    context
}

/// Lowers `src` all the way to the `llvm` dialect (the real pipeline's own
/// stage order, `pipeline.rs::emit_object`, minus the vectorization-
/// specific stages this rewrite doesn't interact with — `--convert-linalg-
/// to-loops` in place of `-to-affine-loops`+`affine-super-vectorize`,
/// the usual simplification for a test that is about
/// allocation rather than vectorization: this file is testing *this*
/// rewrite, not vectorization), then runs `cleave-unify-tensor-allocations`.
fn build_unified_module<'c>(context: &'c Context, src: &str) -> cleave_mlir_shim::mlir::ir::Module<'c> {
    let (result, _sources) = compile(vec![("test.cleave".to_string(), src.to_string())], &[]);
    let program = result.unwrap_or_else(|e| panic!("compile failed: {e:?}"));
    let registry = Registry::build(&program);
    if let Err(diags) = check_type_errors(&program, &registry) {
        panic!("type check failed: {diags:?}");
    }
    let units = collect_units(&program, &registry);
    let cps_program = convert_program(units, None);
    let mlir_types = collect_mlir_types(&program);
    let struct_schemas = collect_struct_schemas(&program);
    let mut module = lower_program(context, &cps_program, &mlir_types, struct_schemas);
    assert!(module.as_operation().verify());

    cleave::pipeline::run_passes(&mut module, "builtin.module(inline,convert-elementwise-to-linalg,linalg-fuse-elementwise-ops)")
        .expect("inline/elementwise-to-linalg/fuse must succeed");


    cleave::pipeline::register_passes();
    cleave::pipeline::run_passes(&mut module, "builtin.module(eliminate-empty-tensors,one-shot-bufferize{bufferize-function-boundaries=true})")
        .expect("one-shot-bufferize must succeed");

    cleave::pipeline::run_passes(&mut module, "builtin.module(ownership-based-buffer-deallocation,buffer-deallocation-simplification,bufferization-lower-deallocations,convert-bufferization-to-memref)")
        .expect("buffer-deallocation must succeed");

    // `--expand-strided-metadata`/`--lower-affine`: needed once a real
    // `memref.subview` with a genuinely non-trivial `strided<...>` layout
    // can appear here (`mlir_lower.rs::copy_nested_array`'s own doc comment
    // has the story).
    // `--convert-to-llvm`, *then* `--finalize-memref-to-llvm`, *then*
    // `--convert-to-llvm` again -- see `tests/user_guide.rs::run_i32`'s own
    // doc comment for the full story (isolated there, on a completely
    // unrelated plain-array case): running `--finalize-memref-to-llvm`
    // once, up front, leaves a genuinely unreconcilable `i64`-to-`index`-
    // to-`i64` round trip behind on ordinary (non-subview) `index`-typed
    // constants; a first `--convert-to-llvm` pass gives those a chance to
    // convert cleanly before `--finalize-memref-to-llvm` ever sees them.
    cleave::pipeline::run_passes(&mut module, "builtin.module(convert-linalg-to-loops,convert-scf-to-cf,canonicalize,expand-strided-metadata,lower-affine,canonicalize,convert-to-llvm,finalize-memref-to-llvm,convert-to-llvm,reconcile-unrealized-casts)")
        .expect("lowering to the llvm dialect must succeed");

    unsafe { cleave_mlir_shim::run_pipeline(module.to_raw(), "builtin.module(cleave-unify-tensor-allocations)", false) }
        .expect("cleave-unify-tensor-allocations");
    assert!(
        module.as_operation().verify(),
        "module failed verification after cleave-unify-tensor-allocations\n{}",
        module.as_operation()
    );
    strip_ciface_wrapper_debug_info(&mut module);

    module
}

fn unified_text(context: &Context, src: &str) -> String {
    build_unified_module(context, src).as_operation().to_string()
}

/// Like `unified_text`, but continues to a real JIT invocation of
/// `main() -> f32` — the decisive check: correct *and* computes the same
/// value, not just verifiable text.
fn run_f32_unified(context: &Context, src: &str) -> f32 {
    let module = build_unified_module(context, src);
    let engine = cleave::pipeline::jit(&module, 2);
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
    }
    let mut result: f32 = -1.0;
    unsafe {
        engine
            .invoke_packed("main", &mut [&mut result as *mut f32 as *mut ()])
            .unwrap_or_else(|e| panic!("JIT invocation failed: {e:?}"));
    }
    result
}

const FREESTANDING_SOURCE: &str = r#"
        use linalg;

        fn main() -> f32 {
            let a: Tensor<f32, 4, 4> = Tensor::<f32, 4, 4>(data: [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0, 16.0]
            ]);
            let b: Tensor<f32, 4, 4> = Tensor::<f32, 4, 4>(data: [[2.0; 4]; 4]);
            let c = Ring::add(a, b);
            let d = Ring::mul(c, c);
            d[1, 2]
        }
        "#;

/// A program with *zero* struct-field-crossing tensors -- the case
/// `mlir_lower.rs::build_tensor_descriptor_value` never touches (nothing to
/// redirect a destination for: `Ring::add`/`Ring::mul` here never reach a
/// struct field), the one this module's own doc comment says is left
/// entirely to plain `malloc`/`free` before this rewrite exists.
#[test]
fn freestanding_tensor_arithmetic_computes_the_right_value_after_unification() {
    let context = context();
    let value = run_f32_unified(&context, FREESTANDING_SOURCE);
    // a[1,2] = row 1 ([5,6,7,8]), col 2 = 7; b[1,2] = 2 -> c[1,2] = 9 -> d[1,2] = 81.
    assert_eq!(value, 81.0);
}

/// Structural half: confirms the rewrite actually fired -- no `malloc`/
/// `free` left anywhere, real `cleave_alloc_rc`/`cleave_release_void`
/// calls in their place.
#[test]
fn freestanding_tensor_arithmetic_uses_cleave_alloc_rc_not_malloc() {
    let context = context();
    let text = unified_text(&context, FREESTANDING_SOURCE);
    assert!(
        !text.contains("@malloc"),
        "expected no remaining malloc calls, got:\n{text}"
    );
    assert!(
        !text.contains("@free"),
        "expected no remaining free calls, got:\n{text}"
    );
    assert!(
        text.contains("cleave_alloc_rc"),
        "expected at least one cleave_alloc_rc call, got:\n{text}"
    );
    assert!(
        text.contains("cleave_release_void"),
        "expected at least one cleave_release_void call, got:\n{text}"
    );
}

/// A program with real structs *and* free-standing tensor arithmetic --
/// `cleave_alloc_rc` is already declared (by the struct), so `retarget_
/// calls`'s own "already declared" branch is exercised for real, not just
/// the "rename the old declaration in place" branch the tests above
/// exercise. Also proves both allocation paths (a struct field written
/// in place, and this module's own free-standing rename)
/// coexist correctly in the same program.
#[test]
fn a_program_with_both_a_struct_and_freestanding_tensor_arithmetic_computes_correctly() {
    let context = context();
    let src = r#"
        use linalg;

        struct Boxed { v: Tensor<f32, 4, 4> }

        fn main() -> f32 {
            let a: Tensor<f32, 4, 4> = Tensor::<f32, 4, 4>(data: [[3.0; 4]; 4]);
            let b: Tensor<f32, 4, 4> = Tensor::<f32, 4, 4>(data: [[4.0; 4]; 4]);
            let boxed = Boxed(v: Ring::sub(a, b));
            let free_standing = Ring::mul(a, b);
            boxed.v[0, 0] + free_standing[0, 0]
        }
        "#;
    let value = run_f32_unified(&context, src);
    // boxed.v[0,0] = 3-4 = -1, free_standing[0,0] = 3*4 = 12 -> 11.
    assert_eq!(value, 11.0);
}
