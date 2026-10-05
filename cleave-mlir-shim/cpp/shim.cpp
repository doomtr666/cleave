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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/SymbolTable.h"
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

// The slots `mlir_lower.rs` allocates for arguments passed by pointer
// (`call_arguments`, marked `cleave.arg_slot`, or `cleave.spawn_arg_slot` for
// a spawned call's) start in their function's entry block, one per call site,
// alive for the whole frame: nanoLM v2's `train_gpt` held 2.6k of them, 1.9 MB
// together, past the 1 MB main thread stack. Two fixes, once loops are blocks
// of their function's own region:
// - every slot the inliner carried out of the entry block (into a loop, where
//   it would be a dynamic allocation) goes back to the entry block's start,
//   with its own size constant;
// - an ordinary call's slot is live from its first use to its last (the store
//   of the argument, then the call, or the loads of an inlined callee), when
//   they share a block: `lifetime.start`/`lifetime.end` there let LLVM's stack
//   coloring give disjoint slots the same storage. A spawned call's slot is
//   read by its task until the caller's `sync`, so it keeps the whole frame.
// The marks are dropped. Unmarked allocations are left alone, and so is a slot
// inside an OpenMP region.
extern "C" void cleaveHoistArgSlots(MlirOperation op) {
  unwrap(op)->walk([](LLVM::LLVMFuncOp f) {
    if (f.getBody().empty())
      return;
    Block &entry = f.getBody().front();
    SmallVector<std::pair<LLVM::AllocaOp, bool>> slots;
    f.walk([&](LLVM::AllocaOp a) {
      bool ordinary = static_cast<bool>(a->removeAttr("cleave.arg_slot"));
      bool spawned = static_cast<bool>(a->removeAttr("cleave.spawn_arg_slot"));
      if ((ordinary || spawned) && a->getParentRegion() == &f.getBody())
        slots.push_back({a, ordinary});
    });
    for (auto [a, ordinary] : slots) {
      if (a->getBlock() != &entry) {
        OpBuilder b(&entry, entry.begin());
        Type sizeTy = a.getArraySize().getType();
        auto one = b.create<LLVM::ConstantOp>(a.getLoc(), sizeTy, b.getIntegerAttr(sizeTy, 1));
        a->setOperand(0, one);
        a->moveAfter(one);
      }
      if (!ordinary)
        continue;
      Block *block = nullptr;
      Operation *first = nullptr, *last = nullptr;
      bool oneBlock = true;
      for (Operation *user : a->getUsers()) {
        if (block && user->getBlock() != block) {
          oneBlock = false;
          break;
        }
        block = user->getBlock();
        if (!first || user->isBeforeInBlock(first))
          first = user;
        if (!last || last->isBeforeInBlock(user))
          last = user;
      }
      if (!first || !oneBlock)
        continue;
      OpBuilder b(first);
      b.create<LLVM::LifetimeStartOp>(a.getLoc(), a.getResult());
      b.setInsertionPointAfter(last);
      b.create<LLVM::LifetimeEndOp>(a.getLoc(), a.getResult());
    }
  });
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

// `spawn` (cleave's `doc/plan-spawn.md`), after bufferization: One-Shot
// Bufferize can't see through an `omp.task` region, so `mlir_lower.rs`
// emits a spawned call as an ordinary call preceded by a call to the marker
// `cleave_spawn_next`, and each wait (`Await`, `Sync`) as a call to a
// `cleave_task_wait(...)` marker whose operands are the buffers to keep
// alive until then. Here, in each function:
//   - the call after each spawn marker moves into an `omp.task`; a scalar
//     result goes through a slot written in the task and read back right
//     before each use (a tensor result is already an out-parameter, written
//     in place by the call);
//   - each wait marker becomes `omp.taskwait`;
//   - a function that spawns is renamed `<name>$tasks`, and `<name>` becomes
//     a wrapper calling it directly inside a parallel region already, or
//     inside a new one (`omp.parallel` around `omp.single`) otherwise: tasks
//     need a team to run on, and opening one per spawning function would
//     nest teams, which libomp serializes.
// The markers' declarations are erased. Returns false on failure.
namespace {

bool isMarkerCall(Operation *op, StringRef prefix) {
  auto call = dyn_cast<func::CallOp>(op);
  return call && call.getCallee().starts_with(prefix);
}

// A slot for a value of type `t`, created at `b`'s insertion point.
Value makeSlot(OpBuilder &b, Location loc, Type t) {
  if (LLVM::isCompatibleType(t)) {
    auto one = b.create<LLVM::ConstantOp>(loc, b.getI64Type(), b.getI64IntegerAttr(1));
    return b.create<LLVM::AllocaOp>(loc, LLVM::LLVMPointerType::get(b.getContext()), t, one);
  }
  return b.create<memref::AllocaOp>(loc, MemRefType::get({}, t));
}

bool slotSupported(Type t) {
  return LLVM::isCompatibleType(t) || isa<IntegerType, FloatType, IndexType>(t);
}

void storeSlot(OpBuilder &b, Location loc, Value v, Value slot) {
  if (isa<LLVM::LLVMPointerType>(slot.getType()))
    b.create<LLVM::StoreOp>(loc, v, slot);
  else
    b.create<memref::StoreOp>(loc, v, slot);
}

Value loadSlot(OpBuilder &b, Location loc, Type t, Value slot) {
  if (isa<LLVM::LLVMPointerType>(slot.getType()))
    return b.create<LLVM::LoadOp>(loc, t, slot);
  return b.create<memref::LoadOp>(loc, slot);
}

// Wraps `call` in an `omp.task`; false (call left in place) if a result has
// a type no slot handles.
bool wrapInTask(func::CallOp call) {
  for (Type t : call.getResultTypes())
    if (!slotSupported(t))
      return false;
  OpBuilder b(call);
  Location loc = call.getLoc();
  SmallVector<Value> slots;
  for (Type t : call.getResultTypes())
    slots.push_back(makeSlot(b, loc, t));
  // Every use of a result, outside the task, reads the slot instead. A wait
  // marker's operands are only there to keep buffers alive until it; the
  // marker becomes `omp.taskwait` right after, operands dropped.
  for (auto [res, slot] : llvm::zip(call.getResults(), slots)) {
    for (OpOperand &use : llvm::make_early_inc_range(res.getUses())) {
      if (isMarkerCall(use.getOwner(), "cleave_task_wait"))
        continue;
      OpBuilder ub(use.getOwner());
      use.set(loadSlot(ub, use.getOwner()->getLoc(), res.getType(), slot));
    }
  }
  auto task = b.create<omp::TaskOp>(loc, omp::TaskOperands{});
  Block *body = b.createBlock(&task.getRegion());
  call->moveBefore(body, body->end());
  OpBuilder tb = OpBuilder::atBlockEnd(body);
  for (auto [res, slot] : llvm::zip(call.getResults(), slots))
    storeSlot(tb, loc, res, slot);
  tb.create<omp::TerminatorOp>(loc);
  return true;
}

// `<name>` becomes `<name>$tasks`, plus a wrapper `<name>` running it on a team.
void wrapInParallelRegion(ModuleOp module, func::FuncOp f) {
  MLIRContext *ctx = module.getContext();
  std::string name = f.getName().str();
  std::string inner = name + "$tasks";
  // Synthesized code, no source line: an unknown location. `f`'s own carries
  // its debug-info subprogram, which LLVM allows on one function only.
  Location loc = UnknownLoc::get(ctx);
  OpBuilder mb(f);
  auto wrapper = mb.create<func::FuncOp>(loc, name, f.getFunctionType());
  // The wrapper takes over everything the function was known by (exported
  // name, C interface, visibility); the body becomes a private function.
  wrapper->setAttrs(f->getAttrs());
  f.setName(inner);
  wrapper.setName(name);
  f->removeAttr("llvm.emit_c_interface");
  SymbolTable::setSymbolVisibility(f, SymbolTable::Visibility::Private);

  for (StringRef callee : {"omp_in_parallel", "cleave_parallel_threads"}) {
    if (!module.lookupSymbol<func::FuncOp>(callee)) {
      OpBuilder db = OpBuilder::atBlockBegin(module.getBody());
      auto decl = db.create<func::FuncOp>(loc, callee, FunctionType::get(ctx, {}, {IntegerType::get(ctx, 32)}));
      decl.setPrivate();
    }
  }

  Block *entry = wrapper.addEntryBlock();
  OpBuilder b = OpBuilder::atBlockEnd(entry);
  SmallVector<Value> args(entry->getArguments().begin(), entry->getArguments().end());
  SmallVector<Type> results(f.getFunctionType().getResults().begin(), f.getFunctionType().getResults().end());
  auto inPar = b.create<func::CallOp>(loc, "omp_in_parallel", TypeRange{IntegerType::get(ctx, 32)});
  auto zero = b.create<arith::ConstantIntOp>(loc, 0, 32);
  auto cond = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, inPar.getResult(0), zero);
  auto ifOp = b.create<scf::IfOp>(loc, TypeRange(results), cond, /*withElseRegion=*/true);
  {
    OpBuilder tb = ifOp.getThenBodyBuilder();
    auto direct = tb.create<func::CallOp>(loc, inner, TypeRange(results), args);
    tb.create<scf::YieldOp>(loc, direct.getResults());
  }
  {
    OpBuilder eb = ifOp.getElseBodyBuilder();
    SmallVector<Value> slots;
    for (Type t : results)
      slots.push_back(makeSlot(eb, loc, t));
    // One thread per physical core unless `OMP_NUM_THREADS` says otherwise
    // (`cleave-rt`'s `cleave_parallel_threads`).
    omp::ParallelOperands clauses;
    clauses.numThreads =
        eb.create<func::CallOp>(loc, "cleave_parallel_threads", TypeRange{IntegerType::get(ctx, 32)}).getResult(0);
    auto par = eb.create<omp::ParallelOp>(loc, clauses);
    Block *pb = eb.createBlock(&par.getRegion());
    OpBuilder pbb = OpBuilder::atBlockEnd(pb);
    auto single = pbb.create<omp::SingleOp>(loc, omp::SingleOperands{});
    Block *sb = pbb.createBlock(&single.getRegion());
    OpBuilder sbb = OpBuilder::atBlockEnd(sb);
    auto call = sbb.create<func::CallOp>(loc, inner, TypeRange(results), args);
    for (auto [res, slot] : llvm::zip(call.getResults(), slots))
      storeSlot(sbb, loc, res, slot);
    sbb.create<omp::TerminatorOp>(loc);
    pbb.setInsertionPointToEnd(pb);
    pbb.create<omp::TerminatorOp>(loc);
    eb.setInsertionPointAfter(par);
    SmallVector<Value> loaded;
    for (auto [t, slot] : llvm::zip(results, slots))
      loaded.push_back(loadSlot(eb, loc, t, slot));
    eb.create<scf::YieldOp>(loc, loaded);
  }
  b.setInsertionPointToEnd(entry);
  b.create<func::ReturnOp>(loc, ifOp.getResults());
}

} // namespace

extern "C" bool cleaveLowerSpawns(MlirOperation op, bool tasks) {
  auto module = dyn_cast<ModuleOp>(unwrap(op));
  if (!module)
    return false;
  module.getContext()->getOrLoadDialect<omp::OpenMPDialect>();
  module.getContext()->getOrLoadDialect<scf::SCFDialect>();

  SmallVector<func::FuncOp> spawning;
  module.walk([&](func::FuncOp f) {
    bool any = false;
    SmallVector<Operation *> markers, waits;
    f.walk([&](Operation *o) {
      if (isMarkerCall(o, "cleave_spawn_next"))
        markers.push_back(o);
      else if (isMarkerCall(o, "cleave_task_wait"))
        waits.push_back(o);
    });
    for (Operation *m : markers) {
      // The spawned call: the next call to the unit the marker names
      // (`cleave_spawn_next(<unit>)`) in its block. Other calls can sit in
      // between (the storage the result goes into, allocated right before).
      StringRef callee = cast<func::CallOp>(m).getCallee();
      callee = callee.drop_front(StringRef("cleave_spawn_next(").size()).drop_back();
      Operation *next = m->getNextNode();
      while (next && !(isa<func::CallOp>(next) && cast<func::CallOp>(next).getCallee() == callee))
        next = next->getNextNode();
      m->erase();
      // Without tasks the call stays where it is (serial elision).
      if (tasks && next && wrapInTask(cast<func::CallOp>(next)))
        any = true;
    }
    for (Operation *w : waits) {
      if (tasks) {
        OpBuilder b(w);
        b.create<omp::TaskwaitOp>(w->getLoc(), omp::TaskwaitOperands{});
      }
      w->erase();
    }
    if (any)
      spawning.push_back(f);
  });
  for (func::FuncOp f : spawning)
    wrapInParallelRegion(module, f);
  // The markers' declarations.
  for (auto f : llvm::make_early_inc_range(module.getOps<func::FuncOp>()))
    if (f.isDeclaration() && (f.getName().starts_with("cleave_spawn_next") || f.getName().starts_with("cleave_task_wait")))
      f.erase();
  return true;
}
