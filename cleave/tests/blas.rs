//! `doc/plan-blas-native.md` §7.2 — the differential oracle that was
//! missing the first time (that attempt's own unexplained precision gap:
//! §0's own "0.9341/0.9019/0.5837" numbers, never root-caused). Deliberately
//! its own small pipeline, mirroring `cleave/tests/dps_rewrite.rs`'s own
//! precedent for the identical reason: a real, JIT-executed check, not just
//! "the verifier didn't complain."
//!
//! No `insert_refcounting` here, matching `mlir_lower.rs::run_i32_from_cps`'s
//! own established precedent for a short-lived, single-invocation test —
//! harmless to leak a handful of small tensors once, at process exit.

use cleave::cps::{collect_mlir_types, collect_struct_schemas, collect_units, convert_program};
use cleave::driver::compile;
use cleave::mlir_lower::lower_program;
use cleave::pipeline::{CodegenOptions, check_type_errors, lower_to_llvm, strip_ciface_wrapper_debug_info};
use cleave::registry::Registry;
use melior::Context;
use melior::dialect::DialectRegistry;
use melior::ir::operation::OperationLike;
use melior::pass;
use melior::utility::{parse_pass_pipeline, register_all_dialects};

fn context() -> Context {
    let dialect_registry = DialectRegistry::new();
    register_all_dialects(&dialect_registry);
    let context = Context::new();
    context.append_dialect_registry(&dialect_registry);
    context.load_all_available_dialects();
    context
}

/// Compiles `src` all the way to a real JIT invocation of `main() -> f32`,
/// mirroring `mlir_lower.rs::run_i32_from_cps`'s own three-stage tensor
/// pipeline (elementwise-to-linalg, one-shot-bufferize with function-
/// boundary rewriting, then the ordinary llvm-dialect tail) exactly — see
/// that function's own doc comments for why each stage is a *separate*
/// `PassManager`, and why the `--convert-to-llvm`/`--finalize-memref-to-
/// llvm`/`--convert-to-llvm` sequence is repeated. Only `cleave_alloc_rc`
/// (real `Tensor` construction) and `cleave_blas_sgemm` (`stdlib/blas/
/// blas.cleave`'s own real extern) are registered — this file's own
/// sources never reach `io`/`dynarray`/refcounting at all.
fn run_f32(context: &Context, src: &str) -> f32 {
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
    assert!(
        module.as_operation().verify(),
        "generated MLIR module failed verification"
    );

    let pass_manager = pass::PassManager::new(context);
    pass_manager.add_pass(pass::linalg::create_convert_elementwise_to_linalg_pass());
    pass_manager
        .run(&mut module)
        .expect("convert-elementwise-to-linalg must succeed");

    let pass_manager = pass::PassManager::new(context);
    pass::bufferization::register_one_shot_bufferize_pass();
    parse_pass_pipeline(
        pass_manager.as_operation_pass_manager(),
        "builtin.module(one-shot-bufferize{bufferize-function-boundaries=true})",
    )
    .expect("failed to parse the one-shot-bufferize pass pipeline");
    pass_manager
        .run(&mut module)
        .expect("one-shot-bufferize must succeed");

    let pass_manager = pass::PassManager::new(context);
    pass_manager.add_pass(pass::linalg::create_convert_linalg_to_loops_pass());
    pass_manager.add_pass(pass::conversion::create_scf_to_control_flow());
    pass_manager.add_pass(pass::transform::create_canonicalizer());
    pass_manager.add_pass(pass::memref::create_expand_strided_metadata_pass());
    pass_manager.add_pass(pass::conversion::create_lower_affine());
    pass_manager.add_pass(pass::transform::create_canonicalizer());
    pass_manager.add_pass(pass::conversion::create_to_llvm());
    pass_manager.add_pass(pass::conversion::create_finalize_mem_ref_to_llvm());
    pass_manager.add_pass(pass::conversion::create_to_llvm());
    pass_manager.add_pass(pass::conversion::create_reconcile_unrealized_casts());
    pass_manager
        .run(&mut module)
        .expect("lowering to the llvm dialect must succeed");
    strip_ciface_wrapper_debug_info(&context, module.as_operation_mut());

    let options = CodegenOptions::default();
    cleave::options::set(options.clone());
    lower_to_llvm(context, &mut module, &options).unwrap_or_else(|e| panic!("lower_to_llvm failed: {e:?}"));

    let engine = melior::ExecutionEngine::new(&module, 2, &[], false, false);
    unsafe {
        engine.register_symbol("cleave_alloc_rc", cleave_rt::cleave_alloc_rc as *mut ());
        engine.register_symbol("cleave_blas_sgemm", cleave_rt::cleave_blas_sgemm as *mut ());
        engine.register_symbol("memrefCopy", cleave_rt::memrefCopy as *mut ());
    }
    let mut out: f32 = f32::NAN;
    unsafe {
        engine
            .invoke_packed("main", &mut [&mut out as *mut f32 as *mut ()])
            .expect("JIT invocation must succeed");
    }
    out
}

/// Absolute correctness, against a hand-computed expected value, no
/// `matmul` involved at all — proves `blas::sgemm` itself is right before
/// ever comparing it against anything else. `A = [[1,2],[3,4]]` (2x2),
/// `B = [[5,6],[7,8]]` (2x2) -> `A@B = [[19,22],[43,50]]`.
#[test]
fn sgemm_computes_the_right_value_against_a_hand_computed_matrix_product() {
    let context = context();
    let src = r#"
        use linalg;
        use blas;
        fn main() -> f32 {
            let a: Tensor<f32,2,2> = Tensor::<f32,2,2>(data: [[1.0,2.0],[3.0,4.0]]);
            let b: Tensor<f32,2,2> = Tensor::<f32,2,2>(data: [[5.0,6.0],[7.0,8.0]]);
            let c: Tensor<f32,2,2> = Tensor::<f32,2,2>(data: [[0.0,0.0],[0.0,0.0]]);
            let r = sgemm(false, false, 1.0, a, b, 0.0, c);
            r[1, 1]
        }
        "#;
    let value = run_f32(&context, src);
    assert_eq!(value, 50.0, "A@B[1,1] should be 50.0 (43,50 second row)");
}

/// The real oracle (`doc/plan-blas-native.md` §7.2): `blas::sgemm` against
/// `MatMul::matmul` on the *same* data, same shape non-trivial enough to
/// exercise real accumulation (4x4, not 2x2) -- the missing check that
/// would have caught the first attempt's own unexplained precision gap.
#[test]
fn sgemm_agrees_with_native_matmul_on_the_same_data() {
    let context = context();
    let src = r#"
        use linalg;
        use blas;
        fn main() -> f32 {
            let a: Tensor<f32,4,4> = Tensor::<f32,4,4>(data: [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0, 16.0]
            ]);
            let b: Tensor<f32,4,4> = Tensor::<f32,4,4>(data: [
                [16.0, 15.0, 14.0, 13.0],
                [12.0, 11.0, 10.0, 9.0],
                [8.0, 7.0, 6.0, 5.0],
                [4.0, 3.0, 2.0, 1.0]
            ]);
            let zero: Tensor<f32,4,4> = Tensor::<f32,4,4>(data: [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0]
            ]);
            let native = matmul(a, b);
            let blas = sgemm(false, false, 1.0, a, b, 0.0, zero);
            let d00 = native[0,0] - blas[0,0];
            let d11 = native[1,1] - blas[1,1];
            let d22 = native[2,2] - blas[2,2];
            let d33 = native[3,3] - blas[3,3];
            let a0 = if d00 < 0.0 { 0.0 - d00 } else { d00 };
            let a1 = if d11 < 0.0 { 0.0 - d11 } else { d11 };
            let a2 = if d22 < 0.0 { 0.0 - d22 } else { d22 };
            let a3 = if d33 < 0.0 { 0.0 - d33 } else { d33 };
            let m01 = if a0 > a1 { a0 } else { a1 };
            let m23 = if a2 > a3 { a2 } else { a3 };
            if m01 > m23 { m01 } else { m23 }
        }
        "#;
    let max_abs_diff = run_f32(&context, src);
    assert!(
        max_abs_diff < 1e-4,
        "blas::sgemm and matmul disagree by {max_abs_diff} on the diagonal -- the exact \
         unexplained gap doc/plan-blas-native.md §0 flagged, now with a real reproduction"
    );
}
