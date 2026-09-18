//! Proves `cleaveExecutionEngineCreateWithTarget`'s own `target_cpu`/
//! `target_features` parameters have a *real* effect on the compiled code
//! -- not just that the shim compiles and links (`doc/backlog.md`'s own
//! entry on why `--target-cpu`/`--target-features` were "wired end to end
//! but have no real effect" before this shim existed).
//!
//! Real vectorizable IR, not a scalar toy: `vector.fma` on `vector<16xf32>`
//! lowers to a single `llvm.intr.fmuladd` on a 512-bit vector value --
//! AVX-512 can hold that in one `zmm` register and emit one `vfmadd*ps`;
//! without it, LLVM's own instruction legalizer must split the 512-bit
//! operation into narrower ($<=$256-bit) pieces. Confirmed by real
//! disassembly (`llvm-objdump`, part of the same `MLIR_SYS_220_PREFIX`
//! toolchain `mlir-sys` itself already requires -- no extra tool
//! dependency), grepping for a real `zmm` register mention -- a raw byte
//! scan for the `0x62` EVEX prefix byte was tried first and rejected: an
//! object file's own symbol/string tables contain enough incidental `0x62`
//! bytes to false-positive even on code with no AVX-512 in it at all,
//! confirmed directly when this test's first version failed on the
//! `x86-64-v2` case.

use std::fs;
use std::process::Command;

const FMA16_MLIR: &str = r#"
module {
  func.func @fma16(%a: vector<16xf32>, %b: vector<16xf32>, %c: vector<16xf32>) -> vector<16xf32> attributes { llvm.emit_c_interface } {
    %r = vector.fma %a, %b, %c : vector<16xf32>
    return %r : vector<16xf32>
  }
}
"#;

/// Lowers `FMA16_MLIR` to the LLVM dialect and dumps a real object file
/// through `cleave_mlir_shim::ExecutionEngine`, built with the given
/// `target_cpu`/`target_features` override (`""` for either means "use
/// `detectHost`'s own default unchanged"). Returns the dumped file's own
/// path (kept on disk -- `disassemble` below re-reads it via `llvm-objdump`).
fn compile_object(context: &melior::Context, target_cpu: &str, target_features: &str) -> std::path::PathBuf {
    use melior::ir::Module;
    use melior::pass::{self, PassManager};

    let mut module = Module::parse(context, FMA16_MLIR).expect("failed to parse probe module");

    // Melior's own strongly-typed pass constructors, not a textual pipeline
    // string -- `parse_pass_pipeline` needs each named pass registered in
    // this process first (the way `mlir-opt`'s own `main()` does for every
    // pass it ships; a bare library-embedding context does not get that for
    // free), confirmed directly (`'convert-vector-to-llvm' does not refer to
    // a registered pass`) rather than assumed. `pipeline.rs`'s own real
    // pipeline already uses exactly this constructor shape for the same
    // reason.
    let pass_manager = PassManager::new(context);
    pass_manager.add_pass(pass::conversion::create_vector_to_llvm());
    pass_manager.add_pass(pass::conversion::create_func_to_llvm());
    pass_manager.add_pass(pass::conversion::create_reconcile_unrealized_casts());
    pass_manager
        .run(&mut module)
        .expect("failed to run the lowering pipeline");

    let dump_path = std::env::temp_dir().join(format!(
        "cleave-mlir-shim-probe-{target_cpu}-{target_features}.o"
    ));
    let engine = cleave_mlir_shim::ExecutionEngine::new(
        module.to_raw(),
        /* optimization_level = */ 2,
        &[],
        /* enable_object_dump = */ true,
        /* enable_pic = */ false,
        target_cpu,
        target_features,
    );
    engine.dump_to_object_file(dump_path.to_str().unwrap());
    assert!(dump_path.exists(), "dumped object file should exist");
    dump_path
}

/// Disassembles `object_path` via `llvm-objdump` (the same `MLIR_SYS_220_
/// PREFIX/bin` toolchain `mlir-sys` itself already needs -- no separate
/// tool dependency), returning the real disassembly text.
fn disassemble(object_path: &std::path::Path) -> String {
    let prefix = std::env::var("MLIR_SYS_220_PREFIX")
        .expect("MLIR_SYS_220_PREFIX must be set to disassemble the probe object");
    let objdump = std::path::Path::new(&prefix).join("bin/llvm-objdump.exe");
    let output = Command::new(&objdump)
        .arg("-d")
        .arg("--x86-asm-syntax=intel")
        .arg(object_path)
        .output()
        .unwrap_or_else(|e| panic!("failed to run {}: {e}", objdump.display()));
    assert!(
        output.status.success(),
        "llvm-objdump failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn disassembly_uses_zmm(object_path: &std::path::Path) -> bool {
    disassemble(object_path).contains("zmm")
}

fn probe_context() -> melior::Context {
    use melior::Context;
    use melior::dialect::DialectRegistry;
    use melior::utility::register_all_dialects;

    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    let context = Context::new();
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
    context.attach_diagnostic_handler(|diagnostic| {
        eprintln!("mlir diagnostic: {diagnostic}");
        true
    });
    context
}

#[test]
fn default_target_uses_the_real_host_cpu() {
    // This test only means anything on a real AVX-512 host -- confirmed
    // directly earlier this session (`bench/fma_roofline`, the same
    // machine): skip rather than give a false negative on a machine without
    // AVX-512 at all.
    if !is_x86_feature_detected!("avx512f") {
        eprintln!("host has no AVX-512 -- skipping (this test needs one to mean anything)");
        return;
    }
    let context = probe_context();
    let path = compile_object(&context, "", "");
    let used_zmm = disassembly_uses_zmm(&path);
    let _ = fs::remove_file(&path);
    assert!(
        used_zmm,
        "default (host-detected) target should use AVX-512 (zmm) on this machine"
    );
}

#[test]
fn target_cpu_native_means_the_max_this_host_actually_has() {
    // Raised directly: an older doc comment claimed `"native"` was "a real,
    // standard LLVM value, resolved by LLVM's own backend at codegen time"
    // -- worth checking directly rather than trusting that, since `native`
    // resolution is normally a *driver*-level convention (clang calls
    // `sys::getHostCPUName()` itself and substitutes the real name *before*
    // the backend ever sees `-mcpu=`), not necessarily something a raw
    // `setCPU("native")` call into `JITTargetMachineBuilder` understands the
    // same way.
    if !is_x86_feature_detected!("avx512f") {
        eprintln!("host has no AVX-512 -- skipping (this test needs one to mean anything)");
        return;
    }
    let context = probe_context();
    let path = compile_object(&context, "native", "");
    let used_zmm = disassembly_uses_zmm(&path);
    let _ = fs::remove_file(&path);
    assert!(
        used_zmm,
        "--target-cpu native should mean \"the real host CPU, AVX-512 included\", \
         but produced no zmm instructions at all"
    );
}

#[test]
fn x86_64_v2_target_cpu_has_a_real_effect_and_drops_avx512() {
    let context = probe_context();
    let path = compile_object(&context, "x86-64-v2", "");
    let used_zmm = disassembly_uses_zmm(&path);
    let _ = fs::remove_file(&path);
    assert!(
        !used_zmm,
        "an explicit x86-64-v2 target (no AVX/AVX2/AVX-512) still produced zmm-register \
         instructions -- the target-cpu override had no real effect"
    );
}

#[test]
fn disabling_avx512_falls_back_to_two_ymm_wide_fma_halves() {
    // Disabling just the one foundation feature AVX-512 depends on doesn't
    // erase FMA itself (a separate feature, `+fma`, available since AVX2)
    // -- LLVM's legalizer should still keep the operation fused, just at
    // half the vector width: the 512-bit `vector<16xf32>` operand no longer
    // fits in one register at all, so it must be split into two genuinely
    // independent 256-bit (`ymm`) `vfmadd*ps` halves instead of one `zmm`
    // one.
    let context = probe_context();
    let path = compile_object(&context, "", "-avx512f,-avx512vl,-avx512bw,-avx512dq,-avx512cd");
    let text = disassemble(&path);
    let _ = fs::remove_file(&path);
    assert!(!text.contains("zmm"), "AVX-512 disabled but zmm still appeared:\n{text}");
    assert!(
        text.contains("ymm"),
        "expected a 256-bit (ymm) fallback with AVX-512 disabled, found none:\n{text}"
    );
    let fma_count = text.matches("vfmadd").count();
    assert!(
        fma_count >= 2,
        "expected the 512-bit fma to split into (at least) two independent 256-bit \
         vfmadd instructions, found {fma_count}:\n{text}"
    );
}

#[test]
fn disabling_fma_alone_also_falls_back_to_ymm_not_just_dropping_the_fusion() {
    // A real surprise, found empirically, not the first guess this test
    // started with (which assumed AVX-512F alone would be enough to keep
    // `zmm` width with a plain, unfused `vmulps`+`vaddps` pair -- it isn't).
    // Disabling *only* `-fma`, with every AVX-512 feature bit still on,
    // still produces a 256-bit (`ymm`) split, not a 512-bit (`zmm`) one --
    // LLVM's own legalization of `llvm.intr.fmuladd` on this backend ties
    // its "preferred vector width" choice to FMA availability itself for
    // this pattern, not to `avx512f` alone. A real, confirmed backend
    // behaviour, not a bug in the shim or this test: the *un*fused
    // `vmulps`/`vaddps` pair this test now asserts on is genuinely
    // representable at `zmm` width without FMA (plain AVX-512F is
    // sufficient for that), so this is LLVM's own choice, not a hard
    // ISA constraint being worked around.
    let context = probe_context();
    let path = compile_object(&context, "", "-fma");
    let text = disassemble(&path);
    let _ = fs::remove_file(&path);
    assert!(
        !text.contains("vfmadd"),
        "-fma was disabled but a fused vfmadd instruction still appeared:\n{text}"
    );
    assert!(!text.contains("zmm"), "expected no zmm usage with -fma disabled:\n{text}");
    assert!(
        text.contains("ymm") && text.contains("vmulps") && text.contains("vaddps"),
        "expected an unfused vmulps + vaddps pair, split to 256-bit (ymm):\n{text}"
    );
}

#[test]
fn explicit_negative_avx512_feature_drops_avx512() {
    // A second, independent way to ask for the same thing as the test
    // above, kept alongside it rather than replacing it: `setCPU("x86-64-
    // v2")` alone might not be enough if `JITTargetMachineBuilder::create
    // TargetMachine`'s own subtarget lookup doesn't resolve that particular
    // microarchitecture-level name the same way `llc -mcpu=` does on the
    // command line -- an explicit `-avx512f` feature string bypasses CPU-
    // name resolution entirely and disables the one feature everything else
    // AVX-512 depends on directly.
    let context = probe_context();
    let path = compile_object(&context, "", "-avx512f");
    let used_zmm = disassembly_uses_zmm(&path);
    let _ = fs::remove_file(&path);
    assert!(
        !used_zmm,
        "an explicit -avx512f feature override still produced zmm-register instructions -- \
         the target-features override had no real effect"
    );
}
