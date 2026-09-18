// The first real cleave shim function -- see `doc/backlog.md`'s own
// "`--target-cpu`/`--target-features` are wired end to end but have no real
// effect" entry for the full story: `mlirExecutionEngineCreate`'s own C API
// (`mlir-c/ExecutionEngine.h`) always builds its `TargetMachine` via
// `JITTargetMachineBuilder::detectHost()`, with no way for a caller to
// override the CPU/features -- confirmed directly by reading this project's
// own vendored `mlir/lib/CAPI/ExecutionEngine/ExecutionEngine.cpp`, not
// assumed. This file is that exact same function, copied and extended with
// two extra parameters (`targetCpu`/`targetFeatures`) applied to the
// `JITTargetMachineBuilder` before it builds the real `TargetMachine` --
// everything else (dialect translation registration, the optimizing
// transformer, `ExecutionEngineOptions`) is unchanged from upstream.
//
// Proves the shim's own build/link/FFI shape on the narrowest possible real
// surface (`doc/backlog.md`'s own recommended sequencing) -- one function,
// no new types, callable from Rust exactly like any other `mlir-sys`
// function, built via `cc`/`build.rs` against the same local LLVM/MLIR
// prefix `mlir-sys` itself already builds against (`MLIR_SYS_220_PREFIX`,
// temporary until `cleave-llvm-redist`'s own prebuilt release replaces the
// local build entirely).

#include "mlir-c/ExecutionEngine.h"
#include "mlir/CAPI/ExecutionEngine.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h"
#include "llvm/ExecutionEngine/Orc/Mangling.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/TargetParser/Host.h"

using namespace mlir;

extern "C" MlirExecutionEngine cleaveExecutionEngineCreateWithTarget(
    MlirModule op, int optLevel, int numPaths,
    const MlirStringRef *sharedLibPaths, bool enableObjectDump,
    bool enablePIC, MlirStringRef targetCpu, MlirStringRef targetFeatures) {
  static bool initOnce = [] {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmParser();
    llvm::InitializeNativeTargetAsmPrinter();
    return true;
  }();
  (void)initOnce;

  auto &ctx = *unwrap(op)->getContext();
  mlir::registerBuiltinDialectTranslation(ctx);
  mlir::registerLLVMDialectTranslation(ctx);
  mlir::registerOpenMPDialectTranslation(ctx);

  auto tmBuilderOrError = llvm::orc::JITTargetMachineBuilder::detectHost();
  if (!tmBuilderOrError) {
    consumeError(tmBuilderOrError.takeError());
    return MlirExecutionEngine{nullptr};
  }
  if (enablePIC)
    tmBuilderOrError->setRelocationModel(llvm::Reloc::PIC_);

  // The lines `mlirExecutionEngineCreate` itself has no way to reach --
  // everything else in this function is an unmodified copy.
  //
  // **`detectHost()` doesn't just pick a CPU name -- it also populates real,
  // explicit `+feature` flags for everything the host actually has**
  // (confirmed directly, not assumed: an isolated probe requesting `setCPU
  // ("x86-64-v2")` alone, with no feature string, still produced `zmm`
  // (AVX-512) instructions in the disassembled object -- `cleave-mlir-shim/
  // tests/target_override.rs`'s own `x86_64_v2_target_cpu_has_a_real_
  // effect_and_drops_avx512` records this exact finding). Subtarget feature
  // resolution applies the CPU's own default feature set first, then layers
  // explicit `+`/`-` feature deltas on top in order -- `detectHost()`'s own
  // already-populated `+avx512f` (etc.) deltas out-rank whatever a *new*
  // CPU's own defaults would otherwise imply, since they were added first
  // and never removed. Any real override (CPU or features) therefore clears
  // that inherited list first, giving the caller a clean slate: the
  // requested CPU's own natural defaults, plus only the feature deltas this
  // call explicitly asks for -- never a silent host-detected leftover.
  llvm::StringRef cpu = unwrap(targetCpu);
  llvm::StringRef features = unwrap(targetFeatures);

  // `"native"` is a *driver*-level convention (clang substitutes the real
  // detected name/features itself, before its own backend ever sees `-mcpu=
  // `), not something the backend's own subtarget lookup understands as a
  // literal string -- confirmed directly, not assumed: passing it straight
  // through to `setCPU` produces a real LLVM diagnostic ("'native' is not a
  // recognized processor for this target (ignoring processor)") followed by
  // a *fatal*, non-catchable `LLVM ERROR` abort building a subtarget that
  // can't even do 64-bit codegen (`cleave-mlir-shim/tests/target_override
  // .rs`'s own `target_cpu_native_means_the_max_this_host_actually_has`
  // found this the hard way). Resolved here instead, the same way a real
  // driver would: substitute the real detected name/feature set before
  // `setCPU`/`setFeatures` below ever see it.
  std::string nativeCpu;
  bool isNative = cpu == "native";
  if (isNative) {
    nativeCpu = llvm::sys::getHostCPUName().str();
    cpu = nativeCpu;
  }

  if (!cpu.empty() || !features.empty())
    tmBuilderOrError->setFeatures("");
  if (!cpu.empty())
    tmBuilderOrError->setCPU(cpu.str());
  if (isNative && features.empty()) {
    // No explicit `targetFeatures` alongside `"native"` -- use the real,
    // full CPUID-detected feature set (not just whatever the resolved CPU
    // *model name*'s own generic defaults imply, which can be a strict
    // subset of what this exact stepping/microcode actually supports).
    for (const auto &entry : llvm::sys::getHostCPUFeatures())
      tmBuilderOrError->getFeatures().AddFeature(entry.getKey(), entry.getValue());
  } else if (!features.empty()) {
    tmBuilderOrError->setFeatures(features);
  }

  auto tmOrError = tmBuilderOrError->createTargetMachine();
  if (!tmOrError) {
    consumeError(tmOrError.takeError());
    return MlirExecutionEngine{nullptr};
  }

  SmallVector<StringRef> libPaths;
  for (unsigned i = 0; i < static_cast<unsigned>(numPaths); ++i)
    libPaths.push_back(unwrap(sharedLibPaths[i]));

  auto transformer = mlir::makeOptimizingTransformer(
      optLevel, /*sizeLevel=*/0, /*targetMachine=*/tmOrError->get());
  ExecutionEngineOptions jitOptions;
  jitOptions.transformer = transformer;
  jitOptions.jitCodeGenOptLevel = static_cast<llvm::CodeGenOptLevel>(optLevel);
  jitOptions.sharedLibPaths = libPaths;
  jitOptions.enableObjectDump = enableObjectDump;
  auto jitOrError = ExecutionEngine::create(unwrap(op), jitOptions,
                                             std::move(tmOrError.get()));
  if (!jitOrError) {
    consumeError(jitOrError.takeError());
    return MlirExecutionEngine{nullptr};
  }
  return wrap(jitOrError->release());
}
