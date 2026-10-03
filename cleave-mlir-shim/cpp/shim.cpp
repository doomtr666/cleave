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
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h"
#include "llvm/ExecutionEngine/Orc/Mangling.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/ModRef.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/TargetParser/Host.h"

using namespace mlir;

// Tells LLVM what the `cleave-rt` allocator entry points *are*, the way clang
// annotates `malloc`/`free` (`allockind`/`allocsize`/`alloc-family`, `noalias`
// return, `inaccessiblemem` effects), so its own store forwarding, memcpy
// elision and alloc/free pairing can apply. Only facts that are true are
// declared: `cleave_alloc_rc`/`cleave_alloc_pool` return fresh, unaliased
// memory; `cleave_release_pool` frees its block unconditionally.
//
// Deliberately **not** declared: `cleave_release`/`cleave_release_void` as
// `free`. They only decrement a refcount and free at zero, so telling LLVM
// they free would let it drop stores to a block another holder still reads
// (measured on `examples/mnist-interop` with the promise made anyway, as a
// ceiling: 52 -> 31 `cleave_alloc_rc` call sites, but semantically wrong).
// A release becomes declarable as `free` only where the block is proven
// unique.
//
// Measured effect of the sound subset on the mnist kernel: 52 -> 46
// `cleave_alloc_rc` sites, 48 -> 34 releases, 18 -> 16 `memcpy` sites.
static void annotateAllocators(llvm::Module &m) {
  // `using namespace mlir` above makes bare `Attribute`/`MemoryEffects`
  // ambiguous against `llvm::` -- spelled out instead.
  using LAttr = llvm::Attribute;
  using LMem = llvm::MemoryEffects;
  auto &c = m.getContext();
  auto markAlloc = [&](llvm::StringRef name, unsigned sizeArg) {
    llvm::Function *f = m.getFunction(name);
    if (!f)
      return;
    f->addFnAttr(LAttr::get(
        c, LAttr::AllocKind,
        uint64_t(llvm::AllocFnKind::Alloc | llvm::AllocFnKind::Uninitialized)));
    f->addFnAttr(LAttr::getWithAllocSizeArgs(c, sizeArg, std::nullopt));
    f->addFnAttr(LAttr::get(c, "alloc-family", "cleave"));
    f->addFnAttr(LAttr::NoUnwind);
    f->addFnAttr(LAttr::WillReturn);
    f->setMemoryEffects(LMem::inaccessibleMemOnly());
    f->addRetAttr(LAttr::NoAlias);
  };
  auto markFree = [&](llvm::StringRef name) {
    llvm::Function *f = m.getFunction(name);
    if (!f)
      return;
    f->addFnAttr(
        LAttr::get(c, LAttr::AllocKind, uint64_t(llvm::AllocFnKind::Free)));
    f->addFnAttr(LAttr::get(c, "alloc-family", "cleave"));
    f->addFnAttr(LAttr::NoUnwind);
    f->addFnAttr(LAttr::WillReturn);
    f->setMemoryEffects(LMem::argMemOnly() | LMem::inaccessibleMemOnly());
    f->addParamAttr(0, LAttr::AllocatedPointer);
  };
  markAlloc("cleave_alloc_rc", 0);
  markAlloc("cleave_alloc_pool", 0);
  markFree("cleave_release_pool");
}

// `mlir::makeOptimizingTransformer` (`mlir/lib/ExecutionEngine/OptUtils.cpp`)
// rebuilt with LLVM's loop unrolling a choice: that function hardcodes
// `PipelineTuningOptions::LoopUnrolling = true`. cleave's loops arrive
// already tiled, vectorized and unrolled where it pays (the matmul schedule,
// `unroll_jam.rs`); LLVM unrolling them again took 55% of `opt -O2` on
// nanoLM's kernel (`LoopUnrollPass`, plus the GVN/LICM work on the unrolled
// code), most of a 3-minute compile. Otherwise identical: the per-module
// default pipeline at the level `optLevel` maps to.
static std::function<llvm::Error(llvm::Module *)>
makeTransformer(unsigned optLevel, bool loopUnroll, llvm::TargetMachine *tm) {
  return [optLevel, loopUnroll, tm](llvm::Module *m) -> llvm::Error {
    llvm::OptimizationLevel level;
    switch (optLevel) {
    case 0: level = llvm::OptimizationLevel::O0; break;
    case 1: level = llvm::OptimizationLevel::O1; break;
    case 2: level = llvm::OptimizationLevel::O2; break;
    default: level = llvm::OptimizationLevel::O3; break;
    }
    llvm::LoopAnalysisManager lam;
    llvm::FunctionAnalysisManager fam;
    llvm::CGSCCAnalysisManager cgam;
    llvm::ModuleAnalysisManager mam;
    llvm::PipelineTuningOptions tuning;
    tuning.LoopUnrolling = loopUnroll;
    tuning.LoopInterleaving = true;
    tuning.LoopVectorization = true;
    tuning.SLPVectorization = true;
    llvm::PassBuilder pb(tm, tuning);
    pb.registerModuleAnalyses(mam);
    pb.registerCGSCCAnalyses(cgam);
    pb.registerFunctionAnalyses(fam);
    pb.registerLoopAnalyses(lam);
    pb.crossRegisterProxies(lam, fam, cgam, mam);
    llvm::ModulePassManager mpm;
    if (level == llvm::OptimizationLevel::O0)
      mpm = pb.buildO0DefaultPipeline(level);
    else
      mpm = pb.buildPerModuleDefaultPipeline(level);
    mpm.run(*m, mam);
    return llvm::Error::success();
  };
}

extern "C" MlirExecutionEngine cleaveExecutionEngineCreateWithTarget(
    MlirModule op, int optLevel, int numPaths,
    const MlirStringRef *sharedLibPaths, bool enableObjectDump,
    bool enablePIC, MlirStringRef targetCpu, MlirStringRef targetFeatures,
    bool loopUnroll) {
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

  auto transformer = makeTransformer(optLevel, loopUnroll, tmOrError->get());
  ExecutionEngineOptions jitOptions;
  jitOptions.transformer = [transformer](llvm::Module *m) -> llvm::Error {
    annotateAllocators(*m);
    return transformer(m);
  };
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

// Rewrites the transcendental `math` ops into polynomial approximations made
// of plain `arith`/`vector` ops (MLIR's `PolynomialApproximation.cpp`), so a
// `math.tanh` on a `vector<1024xf32>` becomes packed AVX-512 arithmetic.
// Without it, `--convert-math-to-llvm` emits `llvm.intr.tanh` on the vector,
// and with no vector math library attached LLVM's backend scalarizes it into
// one libm call per element: `tanhf` was 26% of a nanoLM training step
// (uProf, `ucrtbase.dll`), the GELU of every block. Not exposed as a pass by
// MLIR (only a test pass is), hence the shim. Accuracy is a few ulp, not
// libm's correctly rounded-ish results.
extern "C" bool cleaveApproximateMath(MlirOperation op) {
  static const llvm::StringRef approximated[] = {
      "tanh", "exp", "expm1", "log", "log1p", "log2", "erf", "erfc"};
  auto selected = [](StringRef name) {
    name.consume_front("math.");
    return llvm::is_contained(approximated, name);
  };
  // Only the selected `math` ops and what their rewrites create: a module-wide
  // `applyPatternsGreedily` would also fold constants and simplify regions
  // across the whole program, work nobody asked for on 200k lines of IR.
  SmallVector<Operation *> ops;
  unwrap(op)->walk([&](Operation *o) {
    if (o->getDialect() && o->getDialect()->getNamespace() == "math" &&
        selected(o->getName().getStringRef()))
      ops.push_back(o);
  });
  if (ops.empty())
    return true;
  RewritePatternSet patterns(unwrap(op)->getContext());
  populateMathPolynomialApproximationPatterns(patterns, selected);
  GreedyRewriteConfig config;
  config.setStrictness(GreedyRewriteStrictness::ExistingAndNewOps);
  config.enableFolding(false);
  return succeeded(applyOpPatternsGreedily(ops, std::move(patterns), config));
}
