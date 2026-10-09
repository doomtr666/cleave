//! Proves `Target`'s CPU and feature list have a *real* effect on the
//! compiled code
//! -- not just that the shim compiles and links (`doc/backlog.md`'s own
//! entry on why `--target-cpu`/`--target-features` were "wired end to end
//! but have no real effect" before this shim existed).
//!
//! Real vectorizable IR, not a scalar toy: `vector.fma` on `vector<16xf32>`
//! lowers to a single `llvm.intr.fmuladd` on a 512-bit vector value --
//! AVX-512 can hold that in one `zmm` register and emit one `vfmadd*ps`;
//! without it, LLVM's own instruction legalizer must split the 512-bit
//! operation into narrower ($<=$256-bit) pieces. Confirmed by real
//! disassembly (`llvm-objdump`, part of the `CLEAVE_LLVM_PREFIX` toolchain
//! the shim builds against -- no extra tool dependency), grepping for a real `zmm` register mention -- a raw byte
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

/// Lowers `FMA16_MLIR` to the LLVM dialect and writes it as an object file
/// (`cleave_mlir::emit_object`) for a target built with the given
/// `target_cpu`/`target_features` override (`""` for either means "use
/// `detectHost`'s own default unchanged"). Returns the dumped file's own
/// path (kept on disk -- `disassemble` below re-reads it via `llvm-objdump`).
fn compile_object(
    context: &cleave_mlir::Context,
    target_cpu: &str,
    target_features: &str,
) -> std::path::PathBuf {
    use cleave_mlir::ir::Module;

    let module = Module::parse(context, FMA16_MLIR).expect("failed to parse probe module");
    // SAFETY: `module` is a valid module, owned here.
    unsafe {
        cleave_mlir::run_pipeline(
            module.to_raw(),
            "builtin.module(convert-vector-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)",
            false,
        )
    }
    .expect("failed to run the lowering pipeline");

    let dump_path = std::env::temp_dir().join(format!(
        "cleave-mlir-probe-{target_cpu}-{target_features}.o"
    ));
    let opt = |v: &str| (!v.is_empty()).then(|| v.to_string());
    let target = cleave_mlir::Target::new(opt(target_cpu).as_deref(), opt(target_features).as_deref(), 2, false, true)
        .expect("failed to build the target");
    // SAFETY: `module` is a valid module, owned here.
    unsafe { cleave_mlir::emit_object(module.to_raw(), &target, dump_path.to_str().unwrap()) }
        .expect("failed to emit the object");
    assert!(dump_path.exists(), "dumped object file should exist");
    dump_path
}

/// Disassembles `object_path` via `llvm-objdump` (`CLEAVE_LLVM_PREFIX/bin`),
/// returning the disassembly text.
fn disassemble(object_path: &std::path::Path) -> String {
    let prefix = std::env::var("CLEAVE_LLVM_PREFIX")
        .expect("CLEAVE_LLVM_PREFIX must be set to disassemble the probe object");
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

fn probe_context() -> cleave_mlir::Context {
    use cleave_mlir::Context;
    use cleave_mlir::dialect::DialectRegistry;
    use cleave_mlir::utility::register_all_dialects;

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
    let path = compile_object(
        &context,
        "",
        "-avx512f,-avx512vl,-avx512bw,-avx512dq,-avx512cd",
    );
    let text = disassemble(&path);
    let _ = fs::remove_file(&path);
    assert!(
        !text.contains("zmm"),
        "AVX-512 disabled but zmm still appeared:\n{text}"
    );
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
    assert!(
        !text.contains("zmm"),
        "expected no zmm usage with -fma disabled:\n{text}"
    );
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

/// A CPU LLVM doesn't know is an error, not a warning and a silent fallback
/// on a generic processor.
#[test]
fn an_unknown_target_cpu_is_an_error() {
    let error = cleave_mlir::Target::new(Some("not-a-cpu"), None, 2, false, true).err();
    assert!(error.is_some_and(|e| e.contains("not-a-cpu")), "an unknown CPU was accepted");
}
