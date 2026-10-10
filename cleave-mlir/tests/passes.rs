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

/// MLIR's own passes run in-process, registered by the shim
/// (`cleaveRegisterPasses`): `ensure-debug-info-scope-on-llvm-func`, which
/// crashed through the earlier Rust bindings (a registration their static
/// link dropped), gives a function its `DISubprogram`.
#[test]
fn the_debug_info_scope_pass_runs_in_process() {
    let source = r#"
      module {
        llvm.func @f(%x: i32) -> i32 {
          llvm.return %x : i32
        } loc("probe.cleave":3:1)
      }
    "#;
    let context = context();
    let module = Module::parse(&context, source).expect("failed to parse the module");
    // SAFETY: `module` is a valid module, owned here.
    unsafe { cleave_mlir::run_pipeline(module.to_raw(), "builtin.module(ensure-debug-info-scope-on-llvm-func)", false) }
        .expect("pipeline failed");
    let location = module.body().first_operation().expect("the function").location().to_string();
    assert!(location.contains("di_subprogram"), "{location}");
}

/// A `linalg` op whose bounds aren't affine dimensions or symbols (a tile's
/// row count computed inside a `scf` loop) becomes `scf.for` loops rather
/// than affine loops the verifier rejects (`'affine.for' op operand cannot be
/// used as a dimension id`), which failed the whole compilation; the others
/// still become affine loops, for the affine passes after them.
#[test]
fn a_linalg_op_with_non_affine_bounds_becomes_scf_loops() {
    let source = r#"
      func.func @f(%m: memref<100x16xf32>, %n: memref<16x16xf32>) {
        %c0 = arith.constant 0 : index
        %c8 = arith.constant 8 : index
        %c100 = arith.constant 100 : index
        %one = arith.constant 1.0 : f32
        scf.parallel (%i) = (%c0) to (%c100) step (%c8) {
          %left = arith.subi %c100, %i : index
          %rows = arith.minsi %left, %c8 : index
          %tile = memref.subview %m[%i, 0] [%rows, 16] [1, 1] : memref<100x16xf32> to memref<?x16xf32, strided<[16, 1], offset: ?>>
          linalg.fill ins(%one : f32) outs(%tile : memref<?x16xf32, strided<[16, 1], offset: ?>>)
          scf.reduce
        }
        linalg.fill ins(%one : f32) outs(%n : memref<16x16xf32>)
        return
      }
    "#;
    let out = run(source, "builtin.module(cleave-lower-non-affine-linalg,convert-linalg-to-affine-loops)");
    assert!(!out.contains("linalg."), "a linalg op is left:\n{out}");
    assert!(out.contains("scf.for"), "the tile should be scf loops:\n{out}");
    assert!(out.contains("affine.for"), "the whole buffer should be affine loops:\n{out}");
}

/// A buffer copied only to change its layout, the copy passed to a function
/// (`f`, then `f` passes it on to `g`) that only reads it: the call gets the
/// buffer itself (`cleave-forward-read-only-arguments`). One where the callee
/// writes its parameter (`w`), and one where the buffer is written between
/// the copy and the call, keep their copies.
#[test]
fn a_layout_only_copy_passed_to_a_reader_is_removed() {
    let module = |callee: &str, between: &str| {
        format!(
            r#"
            func.func private @g(%x: memref<4x4xf32>) -> f32 {{
              %c0 = arith.constant 0 : index
              %v = memref.load %x[%c0, %c0] : memref<4x4xf32>
              return %v : f32
            }}
            func.func private @f(%x: memref<4x4xf32>) -> f32 {{
              %v = call @g(%x) : (memref<4x4xf32>) -> f32
              return %v : f32
            }}
            func.func private @w(%x: memref<4x4xf32>) -> f32 {{
              %one = arith.constant 1.0 : f32
              linalg.fill ins(%one : f32) outs(%x : memref<4x4xf32>)
              return %one : f32
            }}
            func.func @main(%b: memref<4x4xf32>) -> f32 {{
              %s = memref.cast %b : memref<4x4xf32> to memref<4x4xf32, strided<[?, ?], offset: ?>>
              %d = memref.alloc() : memref<4x4xf32>
              memref.copy %s, %d : memref<4x4xf32, strided<[?, ?], offset: ?>> to memref<4x4xf32>
              {between}
              %v = call @{callee}(%d) : (memref<4x4xf32>) -> f32
              memref.dealloc %d : memref<4x4xf32>
              return %v : f32
            }}
            "#
        )
    };
    let pipeline = "builtin.module(cleave-forward-read-only-arguments)";
    let out = run(&module("f", ""), pipeline);
    assert!(!out.contains("memref.copy"), "the copy should go:\n{out}");
    assert!(out.contains("call @f(%arg0)"), "`f` should read the buffer itself:\n{out}");

    let out = run(&module("w", ""), pipeline);
    assert!(out.contains("memref.copy"), "`w` writes its parameter: the copy stays:\n{out}");

    let write = "%z = arith.constant 0.0 : f32\n linalg.fill ins(%z : f32) outs(%b : memref<4x4xf32>)";
    let out = run(&module("f", write), pipeline);
    assert!(out.contains("memref.copy"), "the buffer is written before the call: the copy stays:\n{out}");
}

/// A loop whose condition is false on its constant initial values (`for i in
/// 0..0`) is its condition region run once: the body, and the call in it,
/// go. The same loop entered with a bound only known at run time stays.
#[test]
fn a_loop_whose_condition_is_false_on_entry_is_dropped() {
    let module = |bound: &str| {
        format!(
            r#"
            func.func private @never_called(i32) -> i32
            func.func @main(%n: i32) -> i32 {{
              %zero = arith.constant 0 : i32
              %one = arith.constant 1 : i32
              %r:2 = scf.while (%i = %zero, %acc = %zero) : (i32, i32) -> (i32, i32) {{
                %more = arith.cmpi slt, %i, {bound} : i32
                scf.condition(%more) %i, %acc : i32, i32
              }} do {{
              ^bb0(%i: i32, %acc: i32):
                %v = func.call @never_called(%i) : (i32) -> i32
                %next = arith.addi %i, %one : i32
                scf.yield %next, %v : i32, i32
              }}
              return %r#1 : i32
            }}
            "#
        )
    };
    let pipeline = "builtin.module(cleave-drop-zero-trip-whiles)";
    let out = run(&module("%zero"), pipeline);
    assert!(!out.contains("scf.while"), "the loop never runs: it should go:\n{out}");
    assert!(!out.contains("call @never_called"), "nothing in the body should be called:\n{out}");

    let out = run(&module("%n"), pipeline);
    assert!(out.contains("scf.while"), "a bound known at run time keeps the loop:\n{out}");
}
