//! cleave's passes, run through `run_pipeline` on hand-written modules.

use cleave_mlir::Context;
use cleave_mlir::dialect::DialectRegistry;
use cleave_mlir::ir::Module;
use cleave_mlir::utility::register_all_dialects;

fn context() -> Context {
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    let context = Context::new();
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
    context
}

/// Runs `pipeline` on `source`, returning the module printed.
fn run(source: &str, pipeline: &str) -> String {
    let context = context();
    let module = Module::parse(&context, source).expect("failed to parse the module");
    // SAFETY: `module` is a valid module, owned here.
    unsafe { cleave_mlir::run_pipeline(module.to_raw(), pipeline, false) }.expect("pipeline failed");
    module.as_operation().to_string()
}

/// `cleave-lower-permuted-transfers` lowers a permuted transfer (a column
/// read, which `convert-vector-to-llvm` can't lower) to scalar loops, and
/// only it: a contiguous read in the same function stays one vector read.
/// Lowering every transfer of the module once one was permuted made
/// mnist's training loop 3x slower.
#[test]
fn only_permuted_transfers_are_lowered_to_scalar_loops() {
    let source = r#"
      func.func @f(%m: memref<16x16xf32>, %i: index) -> (vector<16xf32>, vector<16xf32>) {
        %pad = arith.constant 0.0 : f32
        %c0 = arith.constant 0 : index
        %row = vector.transfer_read %m[%i, %c0], %pad {in_bounds = [true]} : memref<16x16xf32>, vector<16xf32>
        %col = vector.transfer_read %m[%c0, %i], %pad {in_bounds = [true], permutation_map = affine_map<(d0, d1) -> (d0)>} : memref<16x16xf32>, vector<16xf32>
        return %row, %col : vector<16xf32>, vector<16xf32>
      }
    "#;
    let out = run(source, "builtin.module(cleave-lower-permuted-transfers)");
    let reads_of_16 = out.lines().filter(|l| l.contains("vector.transfer_read") && l.ends_with("vector<16xf32>")).count();
    assert_eq!(reads_of_16, 1, "the row read should stay one vector read, the column one go:\n{out}");
    assert!(!out.contains("affine_map<(d0, d1) -> (d0)>"), "the column read is still there:\n{out}");
    assert!(out.contains("scf.for"), "the column read should be a scalar loop:\n{out}");
}
