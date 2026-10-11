// cleave's C API over MLIR and LLVM (`doc/plan-mlir-shim.md`): the target and
// code generation (`cleaveTarget*`, `cleaveEmitObject`, `cleaveJitCreate`),
// and the IR rewrites cleave's pipeline runs (`pipeline.rs`). Only handles,
// integers and strings cross it.

#include <cstring>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <vector>
#include "mlir-c/ExecutionEngine.h"
#include "mlir/CAPI/ExecutionEngine.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Utils/GPUUtils.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/Inliner.h"
#include "mlir/Transforms/InliningUtils.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Target/SPIRV/Serialization.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
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
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Transform/IR/Utils.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/Support/SourceMgr.h"

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

// A target: the machine code generation is for, and how it optimizes. Built
// once from cleave's options (`CodegenOptions`), used for both the object
// (`cleaveEmitObject`) and the JIT (`cleaveJitCreate`); each builds its own
// `TargetMachine` from it, since the JIT takes ownership of one.
struct CleaveTarget {
  llvm::orc::JITTargetMachineBuilder builder;
  unsigned optLevel;
  bool loopUnroll;
};

// Reports an error message to the caller, which keeps a copy.
using CleaveErrorCallback = void (*)(const char *, size_t, void *);
static void report(CleaveErrorCallback onError, void *userData,
                   const std::string &message) {
  if (onError)
    onError(message.data(), message.size(), userData);
}

static void initNativeTarget() {
  static bool once = [] {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmParser();
    llvm::InitializeNativeTargetAsmPrinter();
    return true;
  }();
  (void)once;
}

// The host's machine, unless `cpu`/`features` say otherwise. `cpu` empty keeps
// the host's; `native` is the host's CPU with every feature it has (a driver
// convention LLVM's backend doesn't know, resolved here as clang does).
// `features` is a comma-separated `+f`/`-f` list. Either one set starts from a
// clean feature list: the host's detected `+feature`s would otherwise outrank
// the chosen CPU's defaults (an `x86-64-v2` object still used AVX-512).
// `optLevel` 0 to 3; `loopUnroll` lets LLVM unroll loops (`makeTransformer`).
// Null on error, reported through `onError`.
extern "C" CleaveTarget *cleaveTargetCreate(MlirStringRef cpuRef,
                                            MlirStringRef featuresRef,
                                            int optLevel, bool pic,
                                            bool loopUnroll,
                                            CleaveErrorCallback onError,
                                            void *userData) {
  initNativeTarget();
  auto builder = llvm::orc::JITTargetMachineBuilder::detectHost();
  if (!builder) {
    report(onError, userData, llvm::toString(builder.takeError()));
    return nullptr;
  }
  llvm::StringRef cpu = unwrap(cpuRef);
  llvm::StringRef features = unwrap(featuresRef);
  std::string nativeCpu;
  bool isNative = cpu == "native";
  if (isNative) {
    nativeCpu = llvm::sys::getHostCPUName().str();
    cpu = nativeCpu;
  }
  // A CPU LLVM doesn't know is the caller's error to report (LLVM itself
  // only warns, then falls back on a generic processor).
  if (!cpu.empty() && !isNative) {
    std::string lookupError;
    const llvm::Target *t = llvm::TargetRegistry::lookupTarget(
        builder->getTargetTriple(), lookupError);
    if (!t) {
      report(onError, userData, lookupError);
      return nullptr;
    }
    std::unique_ptr<llvm::MCSubtargetInfo> sti(t->createMCSubtargetInfo(
        builder->getTargetTriple(), "generic", ""));
    if (!sti || !sti->isCPUStringValid(cpu)) {
      report(onError, userData, "unknown target CPU `" + cpu.str() + "`");
      return nullptr;
    }
  }
  if (!cpu.empty() || !features.empty())
    builder->setFeatures("");
  if (!cpu.empty())
    builder->setCPU(cpu.str());
  if (isNative && features.empty()) {
    for (const auto &entry : llvm::sys::getHostCPUFeatures())
      builder->getFeatures().AddFeature(entry.getKey(), entry.getValue());
  } else if (!features.empty()) {
    builder->setFeatures(features);
  }
  if (pic)
    builder->setRelocationModel(llvm::Reloc::PIC_);
  builder->setCodeGenOptLevel(static_cast<llvm::CodeGenOptLevel>(optLevel));
  // Checked now rather than at first use: a bad feature list is the
  // caller's error too.
  auto probe = builder->createTargetMachine();
  if (!probe) {
    report(onError, userData, llvm::toString(probe.takeError()));
    return nullptr;
  }
  return new CleaveTarget{std::move(*builder), static_cast<unsigned>(optLevel),
                          loopUnroll};
}

extern "C" void cleaveTargetDestroy(CleaveTarget *target) { delete target; }

// Every dialect translation to LLVM IR cleave's modules use.
static void registerTranslations(MLIRContext &ctx) {
  mlir::registerBuiltinDialectTranslation(ctx);
  mlir::registerLLVMDialectTranslation(ctx);
  mlir::registerOpenMPDialectTranslation(ctx);
}

// Writes `module` (in the LLVM dialect) as an object file at `path`: translated
// to LLVM IR, optimized at the target's level (`makeTransformer`, with the
// allocator annotations), compiled by the target's `TargetMachine`. No JIT:
// the object's external symbols are left for the linker. `false` on error,
// reported through `onError`.
extern "C" bool cleaveEmitObject(MlirModule module, CleaveTarget *target,
                                 MlirStringRef pathRef,
                                 CleaveErrorCallback onError, void *userData) {
  initNativeTarget();
  auto tm = target->builder.createTargetMachine();
  if (!tm) {
    report(onError, userData, llvm::toString(tm.takeError()));
    return false;
  }
  Operation *op = unwrap(module).getOperation();
  registerTranslations(*op->getContext());
  llvm::LLVMContext llvmContext;
  std::unique_ptr<llvm::Module> llvmModule =
      mlir::translateModuleToLLVMIR(op, llvmContext);
  if (!llvmModule) {
    report(onError, userData, "failed to translate the module to LLVM IR");
    return false;
  }
  llvmModule->setDataLayout((*tm)->createDataLayout());
  llvmModule->setTargetTriple((*tm)->getTargetTriple());
  annotateAllocators(*llvmModule);
  if (llvm::Error e = makeTransformer(target->optLevel, target->loopUnroll,
                                      tm->get())(llvmModule.get())) {
    report(onError, userData, llvm::toString(std::move(e)));
    return false;
  }
  std::error_code ec;
  llvm::raw_fd_ostream out(unwrap(pathRef), ec, llvm::sys::fs::OF_None);
  if (ec) {
    report(onError, userData,
           "cannot write `" + unwrap(pathRef).str() + "`: " + ec.message());
    return false;
  }
  llvm::legacy::PassManager codegen;
  if ((*tm)->addPassesToEmitFile(codegen, out, nullptr,
                                 llvm::CodeGenFileType::ObjectFile)) {
    report(onError, userData, "the target can't emit an object file");
    return false;
  }
  codegen.run(*llvmModule);
  out.flush();
  return true;
}

// A JIT for `module` (in the LLVM dialect), compiled for `target`, loading the
// `numPaths` shared libraries at `sharedLibPaths` for the symbols they define.
// Null on error, reported through `onError`.
extern "C" MlirExecutionEngine
cleaveJitCreate(MlirModule module, CleaveTarget *target, int numPaths,
                const MlirStringRef *sharedLibPaths,
                CleaveErrorCallback onError, void *userData) {
  initNativeTarget();
  auto tm = target->builder.createTargetMachine();
  if (!tm) {
    report(onError, userData, llvm::toString(tm.takeError()));
    return MlirExecutionEngine{nullptr};
  }
  registerTranslations(*unwrap(module)->getContext());
  SmallVector<StringRef> libPaths;
  for (int i = 0; i < numPaths; ++i)
    libPaths.push_back(unwrap(sharedLibPaths[i]));
  auto transformer =
      makeTransformer(target->optLevel, target->loopUnroll, tm->get());
  ExecutionEngineOptions jitOptions;
  jitOptions.transformer = [transformer](llvm::Module *m) -> llvm::Error {
    annotateAllocators(*m);
    return transformer(m);
  };
  jitOptions.jitCodeGenOptLevel =
      static_cast<llvm::CodeGenOptLevel>(target->optLevel);
  jitOptions.sharedLibPaths = libPaths;
  auto jit = ExecutionEngine::create(unwrap(module), jitOptions, std::move(*tm));
  if (!jit) {
    report(onError, userData, llvm::toString(jit.takeError()));
    return MlirExecutionEngine{nullptr};
  }
  return wrap(jit->release());
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
static void cleaveHoistArgSlots(MlirOperation op) {
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
      // Through the addresses computed from the slot too: a field of a value
      // living in a slot (`mlir_lower.rs`'s `homes`) is passed on as a
      // `getelementptr` of it, and a lifetime ending at the
      // `getelementptr` freed the slot under the call reading the field.
      SmallVector<Operation *> users;
      SmallVector<Value> addresses{a.getResult()};
      while (!addresses.empty()) {
        Value address = addresses.pop_back_val();
        for (Operation *user : address.getUsers()) {
          users.push_back(user);
          if (auto gep = dyn_cast<LLVM::GEPOp>(user); gep && gep.getBase() == address)
            addresses.push_back(gep.getResult());
        }
      }
      for (Operation *user : users) {
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

// MLIR's inliner inlines every call it legally can. Past a size that gains
// nothing (a call costs nothing next to thousands of operations) and costs a
// lot: LLVM's passes are superlinear in a function's size, and a function
// that inlines the walks over a whole model (initialization, restore,
// clipping, checkpointing: one block of code per leaf) becomes one LLVM
// can't digest. Each call whose callee would bring more than `threshold`
// operations (its body plus, recursively, everything it inlines itself) is
// marked `no_inline`, the attribute the inliner already honours (`spawn`'s
// calls). Small functions, the elementwise operations fusion needs, stay
// inlined. Returns the number of calls marked.
static int64_t cleaveLimitInlining(MlirOperation op, int64_t threshold) {
  auto module = cast<ModuleOp>(unwrap(op));
  const int64_t cap = int64_t(1) << 40;
  std::map<std::string, int64_t> own;
  std::map<std::string, std::vector<std::string>> inlinedCallees;
  module.walk([&](func::FuncOp f) {
    if (f.isExternal())
      return;
    int64_t n = 0;
    f.walk([&](Operation *) { ++n; });
    std::string name = f.getName().str();
    own[name] = n;
    auto &callees = inlinedCallees[name];
    f.walk([&](func::CallOp call) {
      if (!call->hasAttr("no_inline"))
        callees.push_back(call.getCallee().str());
    });
  });
  std::map<std::string, int64_t> sizes;
  std::set<std::string> active;
  std::function<int64_t(const std::string &)> inlinedSize = [&](const std::string &name) -> int64_t {
    if (auto it = sizes.find(name); it != sizes.end())
      return it->second;
    auto body = own.find(name);
    if (body == own.end() || !active.insert(name).second)
      return 0;
    int64_t total = body->second;
    for (const std::string &callee : inlinedCallees[name])
      total = std::min(cap, total + inlinedSize(callee));
    active.erase(name);
    sizes[name] = total;
    return total;
  };
  int64_t marked = 0;
  module.walk([&](func::CallOp call) {
    if (call->hasAttr("no_inline") || inlinedSize(call.getCallee().str()) <= threshold)
      return;
    call->setAttr("no_inline", UnitAttr::get(call.getContext()));
    ++marked;
  });
  // The same decision must reach LLVM, whose own inliner would otherwise
  // inline these functions back (a call's `no_inline` doesn't survive the
  // conversion to the LLVM dialect, and neither did a `#[no_inline]`
  // function's): marked here, turned into the `llvm.func`'s `no_inline`
  // once converted (`cleaveApplyNoInline`).
  module.walk([&](func::FuncOp f) {
    if (!f.isExternal() && (f->hasAttr("no_inline") || inlinedSize(f.getName().str()) > threshold))
      f->setAttr("cleave.noinline", UnitAttr::get(f.getContext()));
  });
  return marked;
}

// A large aggregate copied from memory to memory goes through a value:
// `%v = load P`, possibly `extractvalue`s down to a part of it, then
// `store %v, Q`. LLVM splits such a load and store into one per scalar, and a
// light struct holding a whole model's tensor descriptors is thousands of
// them: nanoLM v2's kernel went from 84k LLVM instructions to 642k in `opt
// -O2`, most of it in a few basic blocks of copies. Each store of a value of
// at least `minBytes` traced back to a load becomes a `memcpy` from the
// loaded address (offset by `getelementptr` to the part), when nothing
// between the load and the store may write the source: only operations
// without memory effects, loads, lifetime markers, and stores into a stack
// slot other than the source's. The source must be a function's argument or
// stack slot, whose only other writers that check can see. Returns the
// number of copies rewritten.
static Value pointerRoot(Value p) {
  while (auto gep = p.getDefiningOp<LLVM::GEPOp>())
    p = gep.getBase();
  return p;
}

static int64_t cleaveCopyAggregatesInMemory(MlirOperation op, int64_t minBytes) {
  auto module = cast<ModuleOp>(unwrap(op));
  DataLayout layout(module);
  SmallVector<LLVM::StoreOp> stores;
  module.walk([&](LLVM::StoreOp s) {
    Type t = s.getValue().getType();
    if (!s.getVolatile_() && isa<LLVM::LLVMStructType, LLVM::LLVMArrayType>(t) &&
        int64_t(layout.getTypeSize(t).getFixedValue()) >= minBytes)
      stores.push_back(s);
  });
  int64_t rewritten = 0;
  for (LLVM::StoreOp s : stores) {
    // `path`: the `extractvalue` positions from the loaded aggregate down to
    // the stored part, outermost first.
    SmallVector<int64_t> path;
    SmallVector<Operation *> chain;
    Value v = s.getValue();
    while (auto ev = v.getDefiningOp<LLVM::ExtractValueOp>()) {
      path.insert(path.begin(), ev.getPosition().begin(), ev.getPosition().end());
      chain.push_back(ev);
      v = ev.getContainer();
    }
    auto load = v.getDefiningOp<LLVM::LoadOp>();
    if (!load || load.getVolatile_() || load->getBlock() != s->getBlock() || !load->isBeforeInBlock(s))
      continue;
    Value source = pointerRoot(load.getAddr());
    auto sourceSlot = source.getDefiningOp<LLVM::AllocaOp>();
    if (!sourceSlot && !isa<BlockArgument>(source))
      continue;
    bool clobbered = false;
    for (Operation *between = load->getNextNode(); between != s.getOperation(); between = between->getNextNode()) {
      if (isMemoryEffectFree(between) || isa<LLVM::LoadOp, LLVM::LifetimeStartOp, LLVM::LifetimeEndOp>(between))
        continue;
      if (auto other = dyn_cast<LLVM::StoreOp>(between)) {
        auto slot = pointerRoot(other.getAddr()).getDefiningOp<LLVM::AllocaOp>();
        if (slot && slot != sourceSlot)
          continue;
      }
      clobbered = true;
      break;
    }
    if (clobbered)
      continue;
    OpBuilder b(s);
    Location loc = s.getLoc();
    auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
    Value from = load.getAddr();
    if (!path.empty()) {
      SmallVector<LLVM::GEPArg> indices{0};
      for (int64_t i : path)
        indices.push_back(int32_t(i));
      from = b.create<LLVM::GEPOp>(loc, ptrTy, load.getType(), from, indices);
    }
    int64_t bytes = layout.getTypeSize(s.getValue().getType()).getFixedValue();
    Value len = b.create<LLVM::ConstantOp>(loc, b.getI64Type(), b.getI64IntegerAttr(bytes));
    b.create<LLVM::MemcpyOp>(loc, s.getAddr(), from, len, false);
    s.erase();
    for (Operation *ev : chain)
      if (ev->use_empty())
        ev->erase();
    if (load->use_empty())
      load.erase();
    ++rewritten;
  }
  return rewritten;
}

// Every `llvm.func` `cleaveLimitInlining` marked `cleave.noinline` (one too
// large to inline, or `#[no_inline]` in the source) gets LLVM's `noinline`,
// so that LLVM's inliner keeps it out of line too: nanoLM's kernel went from
// 84k LLVM instructions to 635k after `opt -O2` without it, and the
// functions MLIR had kept out (the model's checkpointing, the optimizer's
// walk) came back into `train_gpt`.
static void cleaveApplyNoInline(MlirOperation op) {
  unwrap(op)->walk([](LLVM::LLVMFuncOp f) {
    if (f->removeAttr("cleave.noinline"))
      f->setAttr(f.getNoInlineAttrName(), UnitAttr::get(f.getContext()));
  });
}

// Two values computed the same way from the same operands: the same SSA
// value, or results of the same side-effect-free operation (same name,
// attributes, result types) on pairwise-same operands. What tells the two
// `h * 8` of a slice's extract and insert apart before CSE has run.
static bool sameValue(Value a, Value b, int depth = 8) {
  if (a == b)
    return true;
  if (depth == 0)
    return false;
  auto ra = dyn_cast<OpResult>(a), rb = dyn_cast<OpResult>(b);
  if (!ra || !rb || ra.getResultNumber() != rb.getResultNumber())
    return false;
  Operation *x = ra.getOwner(), *y = rb.getOwner();
  if (x->getName() != y->getName() || x->getAttrDictionary() != y->getAttrDictionary() ||
      x->getResultTypes() != y->getResultTypes() || x->getNumOperands() != y->getNumOperands() ||
      x->getNumRegions() != 0 || y->getNumRegions() != 0 || !isMemoryEffectFree(x))
    return false;
  for (auto [p, q] : llvm::zip(x->getOperands(), y->getOperands()))
    if (!sameValue(p, q, depth - 1))
      return false;
  return true;
}

// The base a memref is a view of: through subviews, casts and reinterpret
// casts, down to its allocation or block argument.
static Value viewRoot(Value v) {
  while (Operation *def = v.getDefiningOp()) {
    if (auto sv = dyn_cast<memref::SubViewOp>(def))
      v = sv.getSource();
    else if (auto c = dyn_cast<memref::CastOp>(def))
      v = c.getSource();
    else if (auto rc = dyn_cast<memref::ReinterpretCastOp>(def))
      v = rc.getSource();
    else if (auto meta = dyn_cast<memref::ExtractStridedMetadataOp>(def))
      v = meta.getSource();
    else
      break;
  }
  return v;
}

// The same region of the same buffer: one value, or subviews of one source
// at offsets, sizes and strides computed the same way.
static bool sameRegion(Value a, Value b) {
  if (a == b)
    return true;
  auto x = a.getDefiningOp<memref::SubViewOp>(), y = b.getDefiningOp<memref::SubViewOp>();
  if (!x || !y || x.getSource() != y.getSource() || x.getType() != y.getType())
    return false;
  auto same = [](ArrayRef<OpFoldResult> l, ArrayRef<OpFoldResult> r) {
    if (l.size() != r.size())
      return false;
    for (auto [p, q] : llvm::zip(l, r)) {
      if (isEqualConstantIntOrValue(p, q))
        continue;
      auto vp = dyn_cast<Value>(p), vq = dyn_cast<Value>(q);
      if (!vp || !vq || !sameValue(vp, vq))
        return false;
    }
    return true;
  };
  return same(x.getMixedOffsets(), y.getMixedOffsets()) && same(x.getMixedSizes(), y.getMixedSizes()) &&
         same(x.getMixedStrides(), y.getMixedStrides());
}

// A block of a buffer handed to something that writes through a pointer (an
// `extern` such as `cleave_blas_sgemm`, `stdlib/blas`'s `Sgemm::sgemm`, given
// `Slice::slice(out, ...)` as destination) and put back where it came from
// (`Slice::update(out, ..., same offsets)`). One-Shot Bufferize can't see
// the write, so it works on a copy:
//
//   %tmp = memref.alloc()
//   memref.copy %S, %tmp          %S: the block, a subview of %B
//   ... the write, through a pointer to %tmp
//   memref.copy %tmp, %T          %T: the same block of %B
//
// When nothing between the two copies touches %B any other way, the copy
// is the block itself: %tmp becomes %S, and both copies and %tmp go. %tmp
// may only be viewed (cast, its pointer or layout read) besides the copies
// and its deallocation; nothing in that block may take a pointer into %B
// other than through %tmp before the copy back (it could alias the write).
// After One-Shot Bufferize, before the deallocation passes. Returns how many
// blocks it rewrote.
static int64_t cleaveElideBlockCopies(MlirOperation op) {
  SmallVector<memref::AllocOp> allocs;
  unwrap(op)->walk([&](memref::AllocOp alloc) { allocs.push_back(alloc); });
  int64_t elided = 0;
  for (memref::AllocOp alloc : allocs) {
    Value tmp = alloc.getResult();
    // `%tmp` and its casts to a fully dynamic layout. Its layout may only be
    // read through those: they read it at run time, from whatever buffer
    // they are a cast of, so they give the block's real strides and offset
    // once it replaces `%tmp`. A layout read of `%tmp` itself is a
    // compile-time constant, `%tmp`'s own (an 8-wide block's row stride of 8
    // where the block's is 16, a zero offset), and would be wrong for the
    // block.
    //
    // Assumed, not checkable here: that no such constant was folded before
    // bufferization either, through a cast of a statically laid out buffer
    // (`to_buffer` to the plain layout, then a cast to a dynamic one: the
    // stride read through the cast folds to the plain layout's). True of
    // the buffer One-Shot Bufferize makes for a `to_buffer` whose own result
    // has a dynamic layout, which is how `Sgemm::sgemm` gets `c`'s.
    auto fullyDynamic = [](Type t) {
      auto layout = dyn_cast<StridedLayoutAttr>(cast<MemRefType>(t).getLayout());
      return layout && ShapedType::isDynamic(layout.getOffset()) &&
             llvm::all_of(layout.getStrides(), ShapedType::isDynamic);
    };
    SmallVector<Value> fromTmpList{tmp};
    SmallVector<Operation *> casts;
    bool ok = true;
    for (Operation *user : tmp.getUsers())
      if (auto castOp = dyn_cast<memref::CastOp>(user)) {
        ok &= fullyDynamic(castOp.getType());
        casts.push_back(castOp);
        fromTmpList.push_back(castOp.getResult());
      }
    llvm::SmallPtrSet<Value, 8> fromTmp(fromTmpList.begin(), fromTmpList.end());
    memref::CopyOp copyIn, copyOut;
    SmallVector<Operation *> deallocs;
    for (Value v : fromTmpList)
      for (Operation *user : v.getUsers()) {
        bool direct = v == tmp;
        if (auto copy = dyn_cast<memref::CopyOp>(user)) {
          auto &slot = fromTmp.contains(copy.getTarget()) ? copyIn : copyOut;
          ok &= !slot;
          slot = copy;
        } else if (direct && isa<memref::DeallocOp>(user)) {
          deallocs.push_back(user);
        } else if (direct && isa<memref::CastOp>(user)) {
          // Checked above.
        } else if (direct || !isa<memref::ExtractStridedMetadataOp, memref::ExtractAlignedPointerAsIndexOp>(user)) {
          ok = false;
        }
      }
    if (!ok || !copyIn || !copyOut || copyIn->getBlock() != copyOut->getBlock() ||
        alloc->getBlock() != copyIn->getBlock() || !copyIn->isBeforeInBlock(copyOut))
      continue;
    Value block = copyIn.getSource();
    if (!sameRegion(block, copyOut.getTarget()))
      continue;
    // `%S` must be usable wherever `%tmp` is.
    Operation *blockDef = block.getDefiningOp();
    if (blockDef && (blockDef->getBlock() != alloc->getBlock() || !blockDef->isBeforeInBlock(alloc)))
      continue;
    // A cast of `%tmp` must still be a valid cast of `%S`.
    for (Operation *cast : casts)
      ok &= memref::CastOp::areCastCompatible(block.getType(), cast->getResult(0).getType());
    if (!ok)
      continue;
    Value root = viewRoot(block);
    auto reachesRoot = [&](Operation *o) {
      for (Value operand : o->getOperands())
        if (isa<BaseMemRefType>(operand.getType()) && !fromTmp.contains(operand) && viewRoot(operand) == root)
          return true;
      return false;
    };
    // Only what may touch memory counts: making a view (the copy back's own
    // subview) doesn't, taking a pointer out of one might.
    auto mayAccess = [](Operation *o) {
      return isa<memref::ExtractAlignedPointerAsIndexOp>(o) || o->getNumRegions() != 0 || !isMemoryEffectFree(o);
    };
    for (Operation *o = copyIn->getNextNode(); ok && o != copyOut.getOperation(); o = o->getNextNode())
      o->walk([&](Operation *inner) { ok &= !(mayAccess(inner) && reachesRoot(inner)); });
    for (Operation &o : *alloc->getBlock()) {
      if (&o == copyOut.getOperation())
        break;
      if (isa<memref::ExtractAlignedPointerAsIndexOp>(o) && reachesRoot(&o))
        ok = false;
    }
    if (!ok)
      continue;
    copyIn.erase();
    copyOut.erase();
    for (Operation *d : deallocs)
      d->erase();
    tmp.replaceAllUsesWith(block);
    alloc.erase();
    ++elided;
  }
  return elided;
}

// The buffer a value is, through casts and through loops that carry a
// buffer and hand it back unchanged (every iteration writes it in place):
// an `scf.while` result whose `scf.condition` forwards a "before" argument
// that the "after" region yields back as it received it, traced to the
// loop's initial operand. What a loop accumulating into a tensor
// (`Slice::update` in a loop) leaves as its result.
static Value carriedBuffer(Value v, int depth = 16) {
  while (depth-- > 0) {
    if (auto castOp = v.getDefiningOp<memref::CastOp>()) {
      v = castOp.getSource();
      continue;
    }
    auto result = dyn_cast<OpResult>(v);
    auto loop = result ? dyn_cast<scf::WhileOp>(result.getOwner()) : scf::WhileOp();
    if (!loop)
      return v;
    unsigned i = result.getResultNumber();
    auto before = dyn_cast<BlockArgument>(loop.getConditionOp().getArgs()[i]);
    if (!before || before.getOwner() != loop.getBeforeBody())
      return v;
    unsigned j = before.getArgNumber();
    Value yielded = loop.getYieldOp().getResults()[j];
    Value back = carriedBuffer(yielded, depth);
    auto after = dyn_cast<BlockArgument>(back);
    if (!after || after.getOwner() != loop.getAfterBody() || after.getArgNumber() != i)
      return v;
    v = loop.getInits()[j];
  }
  return v;
}

// A fresh buffer written and then copied whole into a destination of the
// same type (the caller's out-parameter for a function's result, once
// `buffer-results-to-out-params` has run, or a field of a returned tuple):
//
//   %tmp = memref.alloc()
//   ... the writes, through %tmp, casts of it, loops carrying it
//   memref.copy %tmp', %D           (%tmp' is %tmp, through those)
//
// When %D is available wherever %tmp is used, nothing else touches %D before
// the copy and the copied value has no other use, the writes may go to %D
// directly: %tmp becomes %D, the copy and %tmp go. The same type, layout
// included, is what keeps this sound: whatever was derived from %tmp's
// layout at compile time (a row stride, a zero offset) holds for %D too.
// Returns how many buffers it forwarded.
static int64_t cleaveForwardCopiesToDestinations(MlirOperation op) {
  SmallVector<memref::CopyOp> copies;
  unwrap(op)->walk([&](memref::CopyOp copy) { copies.push_back(copy); });
  int64_t forwarded = 0;
  for (memref::CopyOp copy : copies) {
    Value source = copy.getSource(), dest = copy.getTarget();
    auto alloc = carriedBuffer(source).getDefiningOp<memref::AllocOp>();
    if (!alloc || alloc.getType() != dest.getType() || alloc->getBlock() != copy->getBlock() ||
        !alloc->isBeforeInBlock(copy))
      continue;
    Value tmp = alloc.getResult();
    // The copied value: not read after the copy (only freed).
    bool ok = llvm::all_of(source.getUsers(), [&](Operation *user) {
      if (user == copy.getOperation() || isa<memref::DeallocOp>(user))
        return true;
      Operation *ancestor = copy->getBlock()->findAncestorOpInBlock(*user);
      return ancestor && ancestor->isBeforeInBlock(copy);
    });
    // `%D` usable wherever `%tmp` is, and `%tmp` used only before the copy.
    auto func = alloc->getParentOfType<FunctionOpInterface>();
    if (!ok || !func)
      continue;
    DominanceInfo dominance(func);
    SmallVector<Operation *> deallocs;
    for (Operation *user : tmp.getUsers()) {
      if (isa<memref::DeallocOp>(user)) {
        deallocs.push_back(user);
        continue;
      }
      Operation *ancestor = alloc->getBlock()->findAncestorOpInBlock(*user);
      ok &= dominance.properlyDominates(dest, user) && ancestor && ancestor->isBeforeInBlock(copy);
    }
    // Nothing touches `%D` between `%tmp`'s allocation and the copy.
    Value root = viewRoot(dest);
    for (Operation *o = alloc->getNextNode(); ok && o != copy.getOperation(); o = o->getNextNode())
      o->walk([&](Operation *inner) {
        for (Value operand : inner->getOperands())
          if (isa<BaseMemRefType>(operand.getType()) && viewRoot(operand) == root)
            ok = false;
      });
    if (!ok)
      continue;
    for (Operation *user : llvm::make_early_inc_range(source.getUsers()))
      if (isa<memref::DeallocOp>(user))
        user->erase();
    copy.erase();
    for (Operation *d : deallocs)
      d->erase();
    tmp.replaceAllUsesWith(dest);
    alloc.erase();
    ++forwarded;
  }
  // The other direction: a fresh buffer whose first write is a whole copy
  // of a buffer of the same type that isn't used afterwards (a BLAS product,
  // its buffer laid out dynamically for `sgemm`, passed to a function whose
  // parameter has the plain layout: One-Shot Bufferize copies it into a
  // fresh plain buffer). The destination becomes that buffer, sound for the
  // same reason: the same type.
  copies.clear();
  unwrap(op)->walk([&](memref::CopyOp copy) { copies.push_back(copy); });
  for (memref::CopyOp copy : copies) {
    Value source = copy.getSource(), dest = copy.getTarget();
    auto alloc = carriedBuffer(source).getDefiningOp<memref::AllocOp>();
    auto fresh = dest.getDefiningOp<memref::AllocOp>();
    if (!alloc || !fresh || alloc.getType() != fresh.getType() || alloc->getBlock() != copy->getBlock() ||
        fresh->getBlock() != copy->getBlock() || !alloc->isBeforeInBlock(copy))
      continue;
    Block *block = copy->getBlock();
    auto after = [&](Operation *user) {
      Operation *ancestor = block->findAncestorOpInBlock(*user);
      return ancestor && copy->isBeforeInBlock(ancestor);
    };
    auto before = [&](Operation *user) {
      Operation *ancestor = block->findAncestorOpInBlock(*user);
      return ancestor && ancestor->isBeforeInBlock(copy);
    };
    // The destination only used after the copy; the source buffer, and the
    // value copied, only before it (or freed).
    bool ok = llvm::all_of(dest.getUsers(), [&](Operation *u) { return u == copy.getOperation() || after(u); });
    for (Value v : SmallVector<Value, 2>{alloc.getResult(), source})
      ok &= llvm::all_of(v.getUsers(), [&](Operation *u) {
        return u == copy.getOperation() || isa<memref::DeallocOp>(u) || before(u);
      });
    if (!ok)
      continue;
    for (Operation *user : llvm::make_early_inc_range(alloc.getResult().getUsers()))
      if (isa<memref::DeallocOp>(user))
        user->erase();
    copy.erase();
    dest.replaceAllUsesWith(alloc.getResult());
    fresh.erase();
    ++forwarded;
  }
  return forwarded;
}

// Whether a function's argument is only read: by itself and through every
// view of it, in the function and in whatever it passes it to. Keyed by
// function and argument number; a function being analyzed counts as
// read-only for the recursion through it (any write found still makes the
// whole answer false).
using ReadOnlyArguments = llvm::DenseMap<std::pair<Operation *, unsigned>, bool>;

static bool valueOnlyRead(Value v, ModuleOp module, ReadOnlyArguments &memo);

static bool argumentOnlyRead(func::FuncOp callee, unsigned index, ModuleOp module, ReadOnlyArguments &memo) {
  if (callee.isExternal() || index >= callee.getNumArguments())
    return false;
  auto key = std::make_pair(callee.getOperation(), index);
  if (auto known = memo.find(key); known != memo.end())
    return known->second;
  memo[key] = true;
  bool onlyRead = valueOnlyRead(callee.getArgument(index), module, memo);
  memo[key] = onlyRead;
  return onlyRead;
}

// Whether the use `use` of a memref only reads it.
static bool useOnlyReads(OpOperand &use, ModuleOp module, ReadOnlyArguments &memo) {
  Operation *user = use.getOwner();
  Value v = use.get();
  if (auto call = dyn_cast<func::CallOp>(user)) {
    auto callee = module.lookupSymbol<func::FuncOp>(call.getCallee());
    return callee && argumentOnlyRead(callee, use.getOperandNumber(), module, memo);
  }
  // A view of it: read only if every use of the view is.
  if (isa<memref::CastOp, memref::SubViewOp, memref::ReinterpretCastOp, memref::CollapseShapeOp,
          memref::ExpandShapeOp, memref::TransposeOp, memref::ExtractStridedMetadataOp>(user))
    return llvm::all_of(user->getResults(), [&](Value result) {
      return !isa<BaseMemRefType>(result.getType()) || valueOnlyRead(result, module, memo);
    });
  if (isa<memref::DimOp>(user))
    return true;
  // What the operation declares it does to this operand: only reads.
  auto effects = dyn_cast<MemoryEffectOpInterface>(user);
  if (!effects)
    return false;
  SmallVector<MemoryEffects::EffectInstance> instances;
  effects.getEffects(instances);
  for (const MemoryEffects::EffectInstance &effect : instances) {
    if (isa<MemoryEffects::Read>(effect.getEffect()))
      continue;
    // A write, an allocation or a free: on this operand, or on what isn't
    // said (any memory).
    if (!effect.getValue() || effect.getValue() == v)
      return false;
  }
  return true;
}

static bool valueOnlyRead(Value v, ModuleOp module, ReadOnlyArguments &memo) {
  return llvm::all_of(v.getUses(), [&](OpOperand &use) { return useOnlyReads(use, module, memo); });
}

// A buffer copied whole into a fresh one only to change its layout, the
// copy then passed to functions that only read it (a BLAS product, laid
// out dynamically for `sgemm` by `cleaveLowerBlasMatmuls`, passed to a
// function whose parameter has the plain layout: One-Shot Bufferize copies
// it, and the source is read again afterwards, so
// `cleaveForwardCopiesToDestinations` keeps the copy):
//
//   %src = memref.cast %buf          (%buf of the plain layout)
//   %d = memref.alloc()              (the plain layout too)
//   memref.copy %src, %d
//   call @f(%d)                      (`f` only reads that argument)
//
// The calls get %buf itself, the copy and %d go: the same memory, read the
// same way, when nothing writes %buf between the copy and the last call.
// Returns how many copies it removed.
// Vulkan backend: a `memref.copy` from one heap payload to another (a
// loop-carried tensor bufferization copies at each iteration, an
// out-parameter's result) made a `linalg.copy`, which becomes a kernel like
// every `linalg` op: the copy stays on the device instead of bringing both
// sides back to the host. Statically shaped memrefs of numbers only; a copy
// involving the stack or a global stays the host's. Returns how many copies
// it rewrote.
using HeapMemo = llvm::DenseMap<std::pair<Operation *, unsigned>, bool>;
static bool isHeapPayload(Value v, ModuleOp module, HeapMemo &memo);

static int64_t cleaveGpuCopies(MlirOperation op) {
  auto module = dyn_cast<ModuleOp>(unwrap(op));
  if (!module)
    return 0;
  HeapMemo heapParams;
  SmallVector<memref::CopyOp> copies;
  module.walk([&](memref::CopyOp copy) {
    auto source = dyn_cast<MemRefType>(copy.getSource().getType());
    auto target = dyn_cast<MemRefType>(copy.getTarget().getType());
    if (source && target && source.hasStaticShape() && target.hasStaticShape() &&
        source.getElementType().isIntOrFloat() && isHeapPayload(copy.getSource(), module, heapParams) &&
        isHeapPayload(copy.getTarget(), module, heapParams))
      copies.push_back(copy);
  });
  for (memref::CopyOp copy : copies) {
    OpBuilder builder(copy);
    linalg::CopyOp::create(builder, copy.getLoc(), copy.getSource(), copy.getTarget());
    copy.erase();
  }
  return copies.size();
}

// Vulkan backend: every `linalg.generic` whose iterators aren't parallel
// ones first, interchanged so that they are. `convert-linalg-to-parallel-
// loops` keeps the iterators' order, so a reduction over rows (`reduce0`,
// `["reduction", "parallel"]`) became a host loop launching one kernel per
// row; interchanged, it is one kernel whose threads each reduce their
// column. Returns how many ops it interchanged.
static int64_t cleaveParallelIteratorsFirst(MlirOperation op) {
  SmallVector<linalg::GenericOp> generics;
  unwrap(op)->walk([&](linalg::GenericOp generic) { generics.push_back(generic); });
  IRRewriter rewriter(unwrap(op)->getContext());
  int64_t interchanged = 0;
  for (linalg::GenericOp generic : generics) {
    SmallVector<utils::IteratorType> iterators = generic.getIteratorTypesArray();
    SmallVector<unsigned> order;
    for (auto [i, kind] : llvm::enumerate(iterators))
      if (kind == utils::IteratorType::parallel)
        order.push_back(i);
    for (auto [i, kind] : llvm::enumerate(iterators))
      if (kind != utils::IteratorType::parallel)
        order.push_back(i);
    if (llvm::equal(order, llvm::seq<unsigned>(0, order.size())))
      continue;
    rewriter.setInsertionPoint(generic);
    if (succeeded(linalg::interchangeGenericOp(rewriter, generic, order)))
      ++interchanged;
  }
  return interchanged;
}

// Vulkan backend (`doc/plan-vulkan.md`). Each outermost `scf.parallel`
// split into workgroups for `gpu-map-parallel-loops` (outer loop: the blocks,
// inner loop: the threads): the dimension indexing its accesses' contiguous
// dimension (else the widest) tiled by `threads` (or its whole extent when
// smaller), every other by 1.
// Whatever rank the loop and in whichever order `linalg` lowered its
// dimensions, a workgroup is then a run of up to `threads` iterations along
// one dimension. A loop whose bounds aren't constants is left as it is.
// Returns how many loops it tiled.
static int64_t cleaveTileParallelLoopsForGpu(MlirOperation op, int64_t threads) {
  SmallVector<scf::ParallelOp> loops;
  unwrap(op)->walk([&](scf::ParallelOp loop) {
    if (!loop->getParentOfType<scf::ParallelOp>())
      loops.push_back(loop);
  });
  int64_t tiled = 0;
  for (scf::ParallelOp loop : loops) {
    SmallVector<int64_t> extents;
    for (auto [lower, upper, step] : llvm::zip(loop.getLowerBound(), loop.getUpperBound(), loop.getStep())) {
      std::optional<int64_t> l = getConstantIntValue(lower), u = getConstantIntValue(upper),
                             st = getConstantIntValue(step);
      if (!l || !u || !st || *st <= 0)
        break;
      extents.push_back((*u - *l + *st - 1) / *st);
    }
    if (extents.size() != loop.getNumLoops())
      continue;
    // The dimension the threads of a workgroup run along: the one that
    // indexes the last (contiguous) dimension of most of the body's memory
    // accesses, so that neighbouring threads touch neighbouring addresses;
    // the widest among those as fast.
    SmallVector<int64_t> contiguous(extents.size(), 0);
    loop.getBody()->walk([&](Operation *access) {
      ValueRange indices;
      if (auto load = dyn_cast<memref::LoadOp>(access))
        indices = load.getIndices();
      else if (auto store = dyn_cast<memref::StoreOp>(access))
        indices = store.getIndices();
      if (indices.empty())
        return;
      Value last = indices.back();
      for (auto [d, iv] : llvm::enumerate(loop.getInductionVars())) {
        auto add = last.getDefiningOp<arith::AddIOp>();
        if (last == iv || (add && (add.getLhs() == iv || add.getRhs() == iv)))
          ++contiguous[d];
      }
    });
    size_t chosen = 0;
    for (size_t d = 1; d < extents.size(); ++d)
      if (contiguous[d] > contiguous[chosen] ||
          (contiguous[d] == contiguous[chosen] && extents[d] > extents[chosen]))
        chosen = d;
    SmallVector<int64_t> sizes(extents.size(), 1);
    sizes[chosen] = std::min(threads, std::max<int64_t>(extents[chosen], 1));
    Operation *parent = loop->getParentOp();
    scf::tileParallelLoop(loop, sizes, /*noMinMaxBounds=*/false);
    // The steps and bounds tiling computes (`step * size`) folded to
    // constants: `convert-parallel-loops-to-gpu` needs a workgroup size it
    // can read, and leaves a loop whose size it can't on the host. Folded
    // here, not by `canonicalize`, which would also merge the two nested
    // loops into one, all of it then mapped to blocks of one thread.
    // `affine.min`'s constant operands folded into its map too, so that the
    // conversion can bound the inner loop by the tile size.
    MLIRContext *ctx = parent->getContext();
    RewritePatternSet patterns(ctx);
    affine::AffineMinOp::getCanonicalizationPatterns(patterns, ctx);
    arith::MulIOp::getCanonicalizationPatterns(patterns, ctx);
    arith::AddIOp::getCanonicalizationPatterns(patterns, ctx);
    SmallVector<Operation *> scalar;
    parent->walk([&](Operation *op) {
      if (isa<arith::MulIOp, arith::AddIOp, arith::SubIOp, affine::AffineMinOp, arith::ConstantOp>(op))
        scalar.push_back(op);
    });
    (void)applyOpPatternsGreedily(scalar, FrozenRewritePatternSet(std::move(patterns)));
    ++tiled;
  }
  return tiled;
}

// Vulkan backend (`doc/plan-vulkan.md`). A kernel's workgroup size as SPIR-V
// wants it (`spirv.entry_point_abi`, which `convert-gpu-to-spirv` requires),
// from the size `gpu-kernel-outlining` found its launches use
// (`known_block_size`). A kernel launched with a size known only at run time
// is an error: a SPIR-V compute entry point fixes its size. Returns how many
// kernels it set.
static int64_t cleaveSetSpirvEntryAbi(MlirOperation op) {
  int64_t set = 0;
  unwrap(op)->walk([&](gpu::GPUFuncOp func) {
    if (!func.isKernel() || func->hasAttr(spirv::getEntryPointABIAttrName()))
      return;
    DenseI32ArrayAttr known = func.getKnownBlockSizeAttr();
    if (!known) {
      func.emitError("a GPU kernel's workgroup size must be known at compile time");
      return;
    }
    func->setAttr(spirv::getEntryPointABIAttrName(),
                  spirv::getEntryPointABIAttr(func.getContext(), known.asArrayRef()));
    ++set;
  });
  return set;
}

// Marks an address the Vulkan backend takes for its runtime's own calls,
// not one escaping to other code (`cleave-sync-host-accesses`).
static constexpr const char *kRuntimeAddress = "cleave.vk_runtime_address";

// Whether the memory of `alloc` (a `memref.alloc`) can only be reached
// through it: every use is an access, a view, a cast, a free, a metadata
// query or a `linalg` op; never stored, passed to a call, or its address
// taken. Then no other memref aliases it.
static bool allocationStaysLocal(Value alloc) {
  SmallVector<Value> pending{alloc};
  while (!pending.empty()) {
    Value v = pending.pop_back_val();
    for (Operation *user : v.getUsers()) {
      if (isa<memref::LoadOp, memref::StoreOp, memref::DeallocOp, memref::DimOp, memref::ExtractStridedMetadataOp,
              linalg::LinalgOp>(user))
        continue;
      if (isa<ViewLikeOpInterface, memref::CastOp>(user)) {
        for (Value result : user->getResults())
          pending.push_back(result);
        continue;
      }
      return false;
    }
  }
  return true;
}

// Whether `index` is the loop's induction variable (or a cast of it).
static bool isInductionVariable(Value index, Value iv) {
  while (auto castOp = index.getDefiningOp<arith::IndexCastOp>())
    index = castOp.getIn();
  return index == iv;
}

// Whether the iterations of `loop` (an `scf.for` carrying nothing) are
// independent: its body calls nothing and declares the effects of every
// operation; every memref it writes is a local allocation
// (`allocationStaysLocal`), accessed only directly (no view) and always
// with the induction variable at one same index position, so that each
// iteration reads and writes its own slice of it. What it only reads may be
// anything. (A local allocation: a `memref.alloc`, or an out-parameter the
// callers allocate fresh for the call.)
static bool isFreshUnaliased(Value v, ModuleOp module, llvm::DenseMap<Value, bool> &memo);

static bool iterationsAreIndependent(scf::ForOp loop, ModuleOp module, llvm::DenseMap<Value, bool> &fresh) {
  // `CLEAVE_DEBUG_PARALLELIZE=1`: why a loop stays sequential, on stderr.
  static const bool debug = [] {
    const char *v = std::getenv("CLEAVE_DEBUG_PARALLELIZE");
    return v && std::string(v) == "1";
  }();
  auto refuse = [&](const char *why) {
    if (debug) {
      auto func = loop->getParentOfType<func::FuncOp>();
      llvm::errs() << "cleave-parallelize-loops: " << (func ? func.getName() : StringRef("?")) << ": " << why << '\n';
    }
    return false;
  };
  if (loop.getNumRegionIterArgs() != 0)
    return refuse("carries values");
  for (Value bound : {loop.getLowerBound(), loop.getUpperBound(), loop.getStep()})
    if (Operation *def = bound.getDefiningOp(); def && loop->isAncestor(def))
      return refuse("bounds defined in the loop");
  Value iv = loop.getInductionVar();
  llvm::DenseSet<Value> written;
  bool ok = true;
  loop.getBody()->walk([&](Operation *inner) {
    if (isa<CallOpInterface>(inner) || isa<func::CallIndirectOp>(inner)) {
      ok = false;
      return WalkResult::interrupt();
    }
    if (auto store = dyn_cast<memref::StoreOp>(inner))
      written.insert(store.getMemRef());
    else if (!isa<memref::LoadOp>(inner) && !isMemoryEffectFree(inner) &&
             !inner->hasTrait<OpTrait::HasRecursiveMemoryEffects>() && !isa<memref::AllocaOp>(inner)) {
      ok = false;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (!ok)
    return refuse("a call or an operation with undeclared effects");
  if (written.empty())
    return refuse("writes nothing");
  for (Value memref : written) {
    // Defined in the loop: private to an iteration.
    if (Operation *def = memref.getDefiningOp(); def && loop->isAncestor(def))
      continue;
    // A local allocation, or an out-parameter the callers allocate fresh
    // (`isFreshUnaliased`), that nothing in the function lets escape.
    if (!isFreshUnaliased(memref, module, fresh))
      return refuse("writes a buffer that isn't a fresh allocation");
    if (!allocationStaysLocal(memref))
      return refuse("writes a buffer that escapes");
    std::optional<unsigned> position;
    loop.getBody()->walk([&](Operation *inner) {
      ValueRange indices;
      Value accessed;
      if (auto load = dyn_cast<memref::LoadOp>(inner))
        accessed = load.getMemRef(), indices = load.getIndices();
      else if (auto store = dyn_cast<memref::StoreOp>(inner))
        accessed = store.getMemRef(), indices = store.getIndices();
      if (accessed != memref)
        return;
      std::optional<unsigned> here;
      for (auto [i, index] : llvm::enumerate(indices))
        if (isInductionVariable(index, iv))
          here = i;
      if (!here || (position && *position != *here))
        ok = false;
      position = here;
    });
    if (!ok)
      return refuse("accesses a written buffer without the induction variable at one position");
  }
  return true;
}

// A stack buffer used as scratch (cleave writes a multi-index access's
// indices into a `memref<2xi32>` alloca, hoisted to the function's entry,
// then reads them back right away) replaced by the values it carries: each
// load becomes the value the nearest store before it in its block wrote at
// the same constant indices, and the buffer goes. It made a loop's
// iterations look dependent (each writes it), and a kernel can't hold one.
// Only when every use is such a store or load.
static void privatizeScratchAllocas(ModuleOp module) {
  SmallVector<memref::AllocaOp> allocas;
  module.walk([&](memref::AllocaOp alloca) { allocas.push_back(alloca); });
  for (memref::AllocaOp alloca : allocas) {
    Value buffer = alloca.getResult();
    auto constantIndices = [](ValueRange indices) {
      return llvm::all_of(indices, [](Value i) { return getConstantIntValue(i).has_value(); });
    };
    auto sameIndices = [](ValueRange a, ValueRange b) {
      return llvm::equal(a, b, [](Value x, Value y) { return getConstantIntValue(x) == getConstantIntValue(y); });
    };
    SmallVector<std::pair<memref::LoadOp, Value>> forwarded;
    bool scratch = true;
    for (Operation *user : buffer.getUsers()) {
      if (auto store = dyn_cast<memref::StoreOp>(user)) {
        if (store.getMemRef() != buffer || !constantIndices(store.getIndices()))
          scratch = false;
        continue;
      }
      auto load = dyn_cast<memref::LoadOp>(user);
      if (!load || !constantIndices(load.getIndices())) {
        scratch = false;
        break;
      }
      Value value;
      for (Operation *before = load->getPrevNode(); before && !value; before = before->getPrevNode())
        if (auto store = dyn_cast<memref::StoreOp>(before);
            store && store.getMemRef() == buffer && sameIndices(store.getIndices(), load.getIndices()))
          value = store.getValueToStore();
      if (!value) {
        scratch = false;
        break;
      }
      forwarded.push_back({load, value});
    }
    if (!scratch)
      continue;
    for (auto [load, value] : forwarded) {
      load.getResult().replaceAllUsesWith(value);
      load.erase();
    }
    for (Operation *user : llvm::make_early_inc_range(buffer.getUsers()))
      user->erase();
    alloca.erase();
  }
}

// A copy of `loop`, just before it, with the calls in its body inlined (and
// the calls they bring, up to a few rounds): the callees `cleave-limit-
// inlining` kept out of line for their size (`nn`'s `row_rrms`, a 256-wide
// reduction unrolled), never a `#[no_inline]` one nor an external one. Null
// if there is no such call, or one couldn't be inlined.
static scf::ForOp cloneWithCallsInlined(scf::ForOp loop, ModuleOp module) {
  auto inlinable = [&](func::CallOp call) -> func::FuncOp {
    auto callee = module.lookupSymbol<func::FuncOp>(call.getCallee());
    if (!callee || callee.isExternal() || callee->hasAttr("no_inline") || callee.getBody().getBlocks().size() != 1)
      return nullptr;
    return callee;
  };
  auto calls = [&](Operation *within) {
    SmallVector<func::CallOp> found;
    within->walk([&](func::CallOp call) { found.push_back(call); });
    return found;
  };
  SmallVector<func::CallOp> first = calls(loop);
  if (first.empty() || llvm::any_of(first, [&](func::CallOp call) { return !inlinable(call); }))
    return nullptr;
  OpBuilder builder(loop);
  auto copy = cast<scf::ForOp>(builder.clone(*loop));
  InlinerInterface interface(module.getContext());
  InlinerConfig config;
  for (int round = 0; round < 4; ++round) {
    SmallVector<func::CallOp> pending = calls(copy);
    if (pending.empty())
      return copy;
    for (func::CallOp call : pending) {
      func::FuncOp callee = inlinable(call);
      if (!callee || failed(inlineCall(interface, config.getCloneCallback(), call, callee, &callee.getBody()))) {
        copy.erase();
        return nullptr;
      }
      call.erase();
    }
  }
  copy.erase();
  return nullptr;
}

// A loop carrying values (`total = total + f(b)`) split in two: what doesn't
// depend on the carried values, `f(b)`, computed by a loop of its own that
// stores each iteration's values in buffers (`parallel`, which may then have
// independent iterations), and the chain that does, replayed by a loop
// reading them back (`sequential`), in the same order: the same operations
// on the same values, so the same result to the bit, while the work is in
// the first loop, when that work holds a loop. `loop` counts from 0 by 1 to
// a constant; the chain is pure operations without regions; what crosses
// from one part to the other is 32-bit integers and floats (an `i1` or a
// 64-bit buffer isn't a storage buffer the kernel's SPIR-V can have). `buffers` and `frees` are the buffers' allocations
// and deallocations, which the caller erases with the loops if it rejects
// the split.
struct CarriedSplit {
  scf::ForOp parallel, sequential;
  SmallVector<Operation *> buffers, frees;
  void erase() {
    for (Operation *op : frees)
      op->erase();
    sequential.erase();
    parallel.erase();
    for (Operation *op : buffers)
      op->erase();
  }
};

static std::optional<CarriedSplit> splitCarriedChain(scf::ForOp loop) {
  if (loop.getNumRegionIterArgs() == 0)
    return std::nullopt;
  std::optional<int64_t> lower = getConstantIntValue(loop.getLowerBound()),
                         upper = getConstantIntValue(loop.getUpperBound()),
                         step = getConstantIntValue(loop.getStep());
  if (!lower || *lower != 0 || !step || *step != 1 || !upper || *upper <= 0)
    return std::nullopt;
  Block *body = loop.getBody();
  // The chain: the operations depending (through their operands, at any
  // depth) on a carried value.
  llvm::DenseSet<Value> carried(body->getArguments().begin() + 1, body->getArguments().end());
  llvm::DenseSet<Operation *> chain;
  for (Operation &op : body->without_terminator()) {
    bool depends = false;
    op.walk([&](Operation *inner) {
      for (Value operand : inner->getOperands())
        if (carried.contains(operand))
          depends = true;
    });
    if (!depends)
      continue;
    if (op.getNumRegions() != 0 || !isPure(&op))
      return std::nullopt;
    chain.insert(&op);
    for (Value result : op.getResults())
      carried.insert(result);
  }
  // What the chain (and the yield) takes from the rest of the body.
  SetVector<Value> crossing;
  auto collect = [&](Operation *user) {
    for (Value operand : user->getOperands()) {
      Operation *def = operand.getDefiningOp();
      if (def && def->getBlock() == body && !chain.contains(def))
        crossing.insert(operand);
    }
  };
  for (Operation *op : chain)
    collect(op);
  collect(body->getTerminator());
  if (crossing.empty() || chain.empty())
    return std::nullopt;
  // Worth it only when an iteration's work is a loop (`log_sum_exp_row`'s
  // over a row): it then reduces what it reads to a few values, the buffers
  // smaller than that. Without one, they are as large as what the work reads.
  bool works = llvm::any_of(body->without_terminator(), [&](Operation &op) {
    if (chain.contains(&op))
      return false;
    bool loops = false;
    op.walk([&](LoopLikeOpInterface) { loops = true; });
    return loops;
  });
  if (!works)
    return std::nullopt;
  for (Value v : crossing)
    if (!isa<IntegerType, FloatType>(v.getType()) || v.getType().getIntOrFloatBitWidth() != 32)
      return std::nullopt;

  OpBuilder builder(loop);
  Location loc = loop.getLoc();
  CarriedSplit split;
  SmallVector<Value> buffers;
  for (Value v : crossing) {
    auto alloc = memref::AllocOp::create(builder, loc, MemRefType::get({*upper}, v.getType()));
    split.buffers.push_back(alloc);
    buffers.push_back(alloc);
  }
  auto asIndex = [&](OpBuilder &b, Value iv) -> Value {
    return iv.getType().isIndex() ? iv : arith::IndexCastOp::create(b, loc, b.getIndexType(), iv).getResult();
  };
  split.parallel = scf::ForOp::create(builder, loc, loop.getLowerBound(), loop.getUpperBound(), loop.getStep(),
                                      ValueRange{}, [&](OpBuilder &b, Location l, Value iv, ValueRange) {
                                        IRMapping mapping;
                                        mapping.map(loop.getInductionVar(), iv);
                                        for (Operation &op : body->without_terminator())
                                          if (!chain.contains(&op))
                                            b.clone(op, mapping);
                                        Value at = asIndex(b, iv);
                                        for (auto [v, buffer] : llvm::zip(crossing, buffers))
                                          memref::StoreOp::create(b, l, mapping.lookup(v), buffer, ValueRange{at});
                                        scf::YieldOp::create(b, l);
                                      });
  split.sequential = scf::ForOp::create(
      builder, loc, loop.getLowerBound(), loop.getUpperBound(), loop.getStep(), loop.getInitArgs(),
      [&](OpBuilder &b, Location l, Value iv, ValueRange args) {
        IRMapping mapping;
        mapping.map(loop.getInductionVar(), iv);
        mapping.map(loop.getRegionIterArgs(), args);
        Value at = asIndex(b, iv);
        for (auto [v, buffer] : llvm::zip(crossing, buffers))
          mapping.map(v, memref::LoadOp::create(b, l, buffer, ValueRange{at}).getResult());
        for (Operation &op : body->without_terminator())
          if (chain.contains(&op))
            b.clone(op, mapping);
        SmallVector<Value> yielded;
        for (Value v : body->getTerminator()->getOperands())
          yielded.push_back(mapping.lookupOrDefault(v));
        scf::YieldOp::create(b, l, yielded);
      });
  for (Value buffer : buffers)
    split.frees.push_back(memref::DeallocOp::create(builder, loc, buffer));
  return split;
}

// Vulkan backend: cleave's counted loops (`for i in 0..n`, an `scf.while`
// after CPS) raised to `scf.for` (MLIR's `UpliftWhileToFor` patterns), and
// the outermost of each nest whose iterations are independent
// (`iterationsAreIndependent`) made an `scf.parallel`: a kernel, one thread
// per iteration, its inner loops sequential in the thread. `stdlib/nn`'s
// element loops (`causal_softmax`'s rows, `rope_rotate`'s) then run on the
// device instead of reading tensors back to the host element by element.
// A loop whose body calls a function is tried again with its calls inlined
// (`cloneWithCallsInlined`), kept so only if that makes it independent; one
// carrying values, split into the work and the chain (`splitCarriedChain`),
// the work parallelized if its iterations are independent.
// Returns how many loops it parallelized.
static int64_t cleaveParallelizeLoops(MlirOperation op) {
  Operation *root = unwrap(op);
  MLIRContext *ctx = root->getContext();
  {
    RewritePatternSet patterns(ctx);
    scf::populateUpliftWhileToForPatterns(patterns);
    SmallVector<Operation *> loops;
    root->walk([&](scf::WhileOp loop) {
      if (!loop->getParentOfType<gpu::GPUModuleOp>())
        loops.push_back(loop);
    });
    GreedyRewriteConfig config;
    config.setStrictness(GreedyRewriteStrictness::ExistingAndNewOps);
    (void)applyOpPatternsGreedily(loops, FrozenRewritePatternSet(std::move(patterns)), config);
  }
  auto module = dyn_cast<ModuleOp>(root);
  if (!module)
    return 0;
  privatizeScratchAllocas(module);
  llvm::DenseMap<Value, bool> fresh;
  SmallVector<scf::ForOp> candidates;
  // The outermost `scf.for`s within `op`, not counting `op` itself.
  auto outermostLoops = [](Operation *op) {
    SmallVector<scf::ForOp> loops;
    op->walk<WalkOrder::PreOrder>([&](Operation *inner) {
      if (inner == op)
        return WalkResult::advance();
      if (isa<gpu::GPUModuleOp, scf::ParallelOp>(inner))
        return WalkResult::skip();
      if (auto loop = dyn_cast<scf::ForOp>(inner)) {
        loops.push_back(loop);
        return WalkResult::skip();
      }
      return WalkResult::advance();
    });
    return loops;
  };
  SmallVector<scf::ForOp> pending = outermostLoops(root);
  std::reverse(pending.begin(), pending.end());
  while (!pending.empty()) {
    scf::ForOp loop = pending.pop_back_val();
    if (iterationsAreIndependent(loop, module, fresh)) {
      candidates.push_back(loop); // its inner loops stay sequential in the thread
      continue;
    }
    scf::ForOp copy = cloneWithCallsInlined(loop, module);
    if (copy && iterationsAreIndependent(copy, module, fresh)) {
      loop.erase();
      fresh.clear(); // its values' addresses may be reused
      candidates.push_back(copy);
      continue;
    }
    if (std::optional<CarriedSplit> split = splitCarriedChain(copy ? copy : loop)) {
      if (iterationsAreIndependent(split->parallel, module, fresh)) {
        loop->replaceAllUsesWith(split->sequential.getResults());
        loop.erase();
        if (copy)
          copy.erase();
        fresh.clear();
        candidates.push_back(split->parallel);
        continue;
      }
      split->erase();
      fresh.clear();
    }
    if (copy) {
      copy.erase();
      fresh.clear(); // its values' addresses may be reused
    }
    SmallVector<scf::ForOp> inner = outermostLoops(loop);
    pending.append(inner.rbegin(), inner.rend());
  }
  IRRewriter rewriter(ctx);
  for (scf::ForOp loop : candidates) {
    if (const char *d = std::getenv("CLEAVE_DEBUG_PARALLELIZE"); d && std::string(d) == "1")
      llvm::errs() << "cleave-parallelize-loops: parallel in " << loop->getParentOfType<func::FuncOp>().getName() << " at "
                   << loop.getLoc() << '\n';
    rewriter.setInsertionPoint(loop);
    Location loc = loop.getLoc();
    Type index = rewriter.getIndexType();
    auto asIndex = [&](Value v) -> Value {
      return v.getType().isIndex() ? v : arith::IndexCastOp::create(rewriter, loc, index, v).getResult();
    };
    auto parallel = scf::ParallelOp::create(rewriter, loc, ValueRange{asIndex(loop.getLowerBound())},
                                            ValueRange{asIndex(loop.getUpperBound())},
                                            ValueRange{asIndex(loop.getStep())});
    Block *body = parallel.getBody();
    rewriter.setInsertionPointToStart(body);
    Value iv = body->getArgument(0);
    Value original = loop.getInductionVar().getType().isIndex()
                         ? iv
                         : arith::IndexCastOp::create(rewriter, loc, loop.getInductionVar().getType(), iv).getResult();
    // The loop's body, its `scf.yield` dropped, before the parallel loop's
    // terminator.
    Block *loopBody = loop.getBody();
    rewriter.eraseOp(loopBody->getTerminator());
    rewriter.inlineBlockBefore(loopBody, body->getTerminator(), ValueRange{original});
    rewriter.eraseOp(loop);
  }
  return candidates.size();
}

// The memref a raw pointer argument was taken from: `stdlib/blas` passes
// `extract_aligned_pointer_as_index` of a view, plus the view's offset in
// bytes, as an `!llvm.ptr`. Searched backwards through the integer
// arithmetic and casts that build it; null if it isn't that shape.
static Value pointerSource(Value ptr) {
  SmallVector<Value> pending{ptr};
  llvm::SmallPtrSet<Operation *, 16> seen;
  while (!pending.empty()) {
    Value v = pending.pop_back_val();
    Operation *def = v.getDefiningOp();
    if (!def || !seen.insert(def).second)
      continue;
    if (auto extract = dyn_cast<memref::ExtractAlignedPointerAsIndexOp>(def))
      return extract.getSource();
    if (isa<LLVM::IntToPtrOp, arith::IndexCastOp, arith::AddIOp>(def))
      for (Value operand : def->getOperands())
        pending.push_back(operand);
  }
  return Value();
}

// Vulkan backend: each `cleave_blas_sgemm` call (`stdlib/blas`, CBLAS's
// `sgemm` on raw pointers) whose transpositions and sizes are constants,
// made `linalg` ops on the memrefs its pointers come from: `C = beta * C`
// (a fill when `beta` is 0, nothing when 1), then `C += alpha * op(A)
// op(B)` as a `linalg.generic` over `(m, n, k)`, the transpositions in its
// indexing maps. They then become GPU kernels like every `linalg` op,
// instead of a host call that brings everything back from the device.
// Returns how many calls it rewrote.
static int64_t cleaveBlasToLinalg(MlirOperation op) {
  SmallVector<func::CallOp> calls;
  unwrap(op)->walk([&](func::CallOp call) {
    if (call.getCallee() == "cleave_blas_sgemm" && call.getNumOperands() == 16)
      calls.push_back(call);
  });
  int64_t rewritten = 0;
  for (func::CallOp call : calls) {
    auto constInt = [&](unsigned i) { return getConstantIntValue(call.getOperand(i)); };
    std::optional<int64_t> transA = constInt(0), transB = constInt(1), m = constInt(2), n = constInt(3), k = constInt(4);
    if (!transA || !transB || !m || !n || !k)
      continue;
    Value a = pointerSource(call.getOperand(6)), b = pointerSource(call.getOperand(9)),
          c = pointerSource(call.getOperand(13));
    auto shapeIs = [](Value v, int64_t rows, int64_t cols) {
      auto type = v ? dyn_cast<MemRefType>(v.getType()) : MemRefType();
      return type && type.getRank() == 2 && type.getElementType().isF32() && type.getDimSize(0) == rows &&
             type.getDimSize(1) == cols;
    };
    if (!shapeIs(a, *transA ? *k : *m, *transA ? *m : *k) || !shapeIs(b, *transB ? *n : *k, *transB ? *k : *n) ||
        !shapeIs(c, *m, *n))
      continue;
    Value alpha = call.getOperand(5), beta = call.getOperand(12);
    std::optional<double> betaValue;
    if (auto constant = beta.getDefiningOp<arith::ConstantFloatOp>())
      betaValue = constant.value().convertToDouble();

    OpBuilder builder(call);
    Location loc = call.getLoc();
    MLIRContext *ctx = call.getContext();
    auto identity2 = AffineMap::getMultiDimIdentityMap(2, ctx);
    SmallVector<utils::IteratorType> parallel2(2, utils::IteratorType::parallel);
    if (betaValue && *betaValue == 0.0) {
      Value zero = arith::ConstantFloatOp::create(builder, loc, builder.getF32Type(), APFloat(0.0f));
      linalg::FillOp::create(builder, loc, ValueRange{zero}, ValueRange{c});
    } else if (!betaValue || *betaValue != 1.0) {
      linalg::GenericOp::create(builder, loc, TypeRange{}, ValueRange{}, ValueRange{c},
                                ArrayRef<AffineMap>{identity2}, parallel2,
                                [&](OpBuilder &b2, Location l, ValueRange args) {
                                  Value scaled = arith::MulFOp::create(b2, l, args[0], beta);
                                  linalg::YieldOp::create(b2, l, scaled);
                                });
    }
    AffineExpr dm, dn, dk;
    bindDims(ctx, dm, dn, dk);
    auto mapA = AffineMap::get(3, 0, *transA ? ArrayRef<AffineExpr>{dk, dm} : ArrayRef<AffineExpr>{dm, dk}, ctx);
    auto mapB = AffineMap::get(3, 0, *transB ? ArrayRef<AffineExpr>{dn, dk} : ArrayRef<AffineExpr>{dk, dn}, ctx);
    auto mapC = AffineMap::get(3, 0, {dm, dn}, ctx);
    SmallVector<utils::IteratorType> iterators{utils::IteratorType::parallel, utils::IteratorType::parallel,
                                               utils::IteratorType::reduction};
    linalg::GenericOp::create(builder, loc, TypeRange{}, ValueRange{a, b}, ValueRange{c},
                              ArrayRef<AffineMap>{mapA, mapB, mapC}, iterators,
                              [&](OpBuilder &b2, Location l, ValueRange args) {
                                Value product = arith::MulFOp::create(b2, l, args[0], args[1]);
                                Value scaled = arith::MulFOp::create(b2, l, product, alpha);
                                Value sum = arith::AddFOp::create(b2, l, args[2], scaled);
                                linalg::YieldOp::create(b2, l, sum);
                              });
    // The pointer arithmetic that fed the call, now dead.
    // Each op once (an operand can appear twice, a constant several times).
    llvm::SetVector<Operation *> pending;
    for (Value operand : call.getOperands())
      if (Operation *def = operand.getDefiningOp())
        pending.insert(def);
    call.erase();
    while (!pending.empty()) {
      Operation *def = pending.pop_back_val();
      if (!def->use_empty() || !isPure(def))
        continue;
      for (Value operand : def->getOperands())
        if (Operation *producer = operand.getDefiningOp())
          pending.insert(producer);
      def->erase();
    }
    ++rewritten;
  }
  return rewritten;
}

// Vulkan backend: the first byte a memref `v` reaches and how many bytes
// it spans, as two `i64`s (`1 + sum((size - 1) * stride)` elements from its
// offset, whatever its layout): the range the runtime keys its device
// mirrors by (`cleave-lower-gpu-launches`, `cleave-sync-host-accesses`).
// `fromBase`: from the start of the buffer it views instead, the offset
// included in the span: how a kernel argument is bound, since the kernel
// applies a static offset itself (`convert-gpu-to-spirv` keeps it in its
// accesses).
static std::pair<Value, Value> memrefSpan(OpBuilder &builder, Location loc, Value v, bool fromBase = false) {
  auto type = cast<MemRefType>(v.getType());
  Type i64 = builder.getI64Type();
  auto constIndex = [&](int64_t c) -> Value { return arith::ConstantIndexOp::create(builder, loc, c); };
  int64_t elementBytes = type.getElementType().isIntOrFloat() ? type.getElementType().getIntOrFloatBitWidth() / 8 : 8;
  auto metadata = memref::ExtractStridedMetadataOp::create(builder, loc, v);
  auto alignedOp = memref::ExtractAlignedPointerAsIndexOp::create(builder, loc, v);
  alignedOp->setAttr(kRuntimeAddress, builder.getUnitAttr());
  Value aligned = alignedOp;
  Value first = fromBase ? aligned
                         : arith::AddIOp::create(builder, loc, aligned,
                                                 arith::MulIOp::create(builder, loc, metadata.getOffset(),
                                                                       constIndex(elementBytes)));
  Value elements = fromBase ? arith::AddIOp::create(builder, loc, constIndex(1), metadata.getOffset()) : constIndex(1);
  for (auto [size, stride] : llvm::zip(metadata.getSizes(), metadata.getStrides())) {
    Value reach = arith::MulIOp::create(builder, loc, arith::SubIOp::create(builder, loc, size, constIndex(1)), stride);
    elements = arith::AddIOp::create(builder, loc, elements, reach);
  }
  Value bytes = arith::MulIOp::create(builder, loc, elements, constIndex(elementBytes));
  return {arith::IndexCastOp::create(builder, loc, i64, first), arith::IndexCastOp::create(builder, loc, i64, bytes)};
}

// Vulkan backend: whether the memory `v` views is a heap payload (a
// `memref.alloc`, which `cleave-unify-tensor-allocations` makes a
// `cleave_alloc_rc` block, whose frees `cleave-rt` reports): only those may
// stay mirrored on the device across launches. Followed through views,
// through what flows into a loop-carried value or a branch's result (every
// initial value and every yielded one), and through the parameters of
// private functions (every call site's argument), and through struct fields
// when the module's descriptors all view heap payloads
// (`descriptorsAreHeapPayloads`); anything else (the stack, a global, a
// public function's parameter) is not one. A
// value being decided is assumed one: every value that reaches it is still
// checked where it enters.
static bool allHeapPayloads(ValueRange values, ModuleOp module, HeapMemo &memo) {
  return llvm::all_of(values, [&](Value v) { return isHeapPayload(v, module, memo); });
}

// What flows into `v` when it's a loop-carried value, a branch's result or a
// function's parameter: a key for the memo and the values, or no key.
static std::optional<std::pair<std::pair<Operation *, unsigned>, SmallVector<Value>>> heapSources(Value v,
                                                                                                 ModuleOp module) {
  SmallVector<Value> sources;
  if (auto result = dyn_cast<OpResult>(v)) {
    Operation *owner = result.getOwner();
    unsigned i = result.getResultNumber();
    if (auto loop = dyn_cast<scf::ForOp>(owner)) {
      sources.push_back(loop.getInitArgs()[i]);
      sources.push_back(loop.getBody()->getTerminator()->getOperand(i));
    } else if (auto loop = dyn_cast<scf::WhileOp>(owner)) {
      sources.push_back(loop.getConditionOp().getArgs()[i]);
    } else if (auto branch = dyn_cast<scf::IfOp>(owner)) {
      sources.push_back(branch.thenYield().getOperand(i));
      if (!branch.getElseRegion().empty())
        sources.push_back(branch.elseYield().getOperand(i));
    } else {
      return std::nullopt;
    }
    return std::make_pair(std::make_pair(owner, i), sources);
  }
  auto arg = cast<BlockArgument>(v);
  Operation *owner = arg.getOwner()->getParentOp();
  unsigned i = arg.getArgNumber();
  // Keyed apart from the owner's results: argument `i` as `i + 1000000`.
  auto key = std::make_pair(owner, i + 1000000 + 1000 * arg.getOwner()->getParent()->getRegionNumber());
  if (auto loop = dyn_cast<scf::ForOp>(owner)) {
    if (i == 0)
      return std::nullopt; // the induction variable
    sources.push_back(loop.getInitArgs()[i - 1]);
    sources.push_back(loop.getBody()->getTerminator()->getOperand(i - 1));
  } else if (auto loop = dyn_cast<scf::WhileOp>(owner)) {
    if (arg.getOwner()->getParent() == &loop.getBefore()) {
      sources.push_back(loop.getInits()[i]);
      sources.push_back(loop.getYieldOp().getOperand(i));
    } else {
      sources.push_back(loop.getConditionOp().getArgs()[i]);
    }
  } else if (auto func = dyn_cast<func::FuncOp>(owner)) {
    if (arg.getOwner() != &func.getBody().front() || func.isPublic())
      return std::nullopt;
    auto uses = SymbolTable::getSymbolUses(func, module);
    if (!uses)
      return std::nullopt;
    for (const SymbolTable::SymbolUse &use : *uses) {
      auto call = dyn_cast<func::CallOp>(use.getUser());
      if (!call)
        return std::nullopt;
      sources.push_back(call.getOperand(i));
    }
  } else {
    return std::nullopt;
  }
  return std::make_pair(key, sources);
}

// Whether `type` is a memref descriptor in the LLVM dialect, as cleave's
// struct fields hold tensors: `(ptr allocated, ptr aligned, i64 offset,
// array<N x i64> sizes, array<N x i64> strides)`, the arrays absent at rank 0.
static bool isMemrefDescriptor(Type type) {
  auto structType = dyn_cast<LLVM::LLVMStructType>(type);
  if (!structType || structType.isPacked())
    return false;
  ArrayRef<Type> body = structType.getBody();
  if (body.size() != 3 && body.size() != 5)
    return false;
  if (!isa<LLVM::LLVMPointerType>(body[0]) || !isa<LLVM::LLVMPointerType>(body[1]) || !body[2].isInteger(64))
    return false;
  return body.size() == 3 || (isa<LLVM::LLVMArrayType>(body[3]) && body[3] == body[4]);
}

// Whether `type` holds a pointer or a memref descriptor anywhere in it.
static bool holdsPointers(Type type) {
  if (isa<LLVM::LLVMPointerType>(type))
    return true;
  if (auto structType = dyn_cast<LLVM::LLVMStructType>(type))
    return llvm::any_of(structType.getBody(), holdsPointers);
  if (auto arrayType = dyn_cast<LLVM::LLVMArrayType>(type))
    return holdsPointers(arrayType.getElementType());
  return false;
}

// The aggregate type a position path leads into last, and the index in it:
// what an `llvm.insertvalue`/`extractvalue` at `position` of `type` writes or
// reads.
static std::pair<Type, int64_t> lastContainer(Type type, ArrayRef<int64_t> position) {
  Type container = type;
  for (int64_t index : position.drop_back()) {
    if (auto structType = dyn_cast<LLVM::LLVMStructType>(container))
      container = structType.getBody()[index];
    else if (auto arrayType = dyn_cast<LLVM::LLVMArrayType>(container))
      container = arrayType.getElementType();
    else
      return {Type(), -1};
  }
  return {container, position.empty() ? -1 : position.back()};
}

static bool isHeapPayload(Value v, ModuleOp module, HeapMemo &memo);

// The runtime's allocations whose frees `cleave-rt` reports
// (`set_free_hook`): a refcounted payload, an arena one (reclaimed in bulk
// at its region's exit, which is reported as a range).
static bool isReportedAllocation(StringRef callee) {
  return callee == "cleave_alloc_rc" || callee == "cleave_alloc_local";
}

// Whether the pointer `p`, stored as a descriptor's allocated or aligned
// pointer, points into a heap payload: an allocation whose free is reported
// (`isReportedAllocation`), a pointer read out of a descriptor
// (`descriptorsAreHeapPayloads` checks every descriptor built), the address
// of a heap payload memref, or null.
static bool isHeapPointer(Value p, ModuleOp module, HeapMemo &memo) {
  Operation *def = p.getDefiningOp();
  if (!def)
    return false;
  if (isa<LLVM::ZeroOp, LLVM::PoisonOp, LLVM::UndefOp>(def))
    return true;
  if (auto call = dyn_cast<func::CallOp>(def))
    return isReportedAllocation(call.getCallee());
  if (auto call = dyn_cast<LLVM::CallOp>(def))
    return call.getCallee() && isReportedAllocation(*call.getCallee());
  if (auto extract = dyn_cast<LLVM::ExtractValueOp>(def)) {
    auto [container, index] = lastContainer(extract.getContainer().getType(), extract.getPosition());
    return container && isMemrefDescriptor(container) && (index == 0 || index == 1);
  }
  if (auto toPtr = dyn_cast<LLVM::IntToPtrOp>(def)) {
    Value integer = toPtr.getArg();
    if (auto castOp = integer.getDefiningOp<arith::IndexCastOp>())
      integer = castOp.getIn();
    if (auto address = integer.getDefiningOp<memref::ExtractAlignedPointerAsIndexOp>())
      return isHeapPayload(address.getSource(), module, memo);
  }
  return false;
}

// Vulkan backend: whether every memref descriptor the module's code can see
// views a heap payload, so that a tensor read out of a struct field is one
// (`isHeapPayload`). A closed-world check, once per module: every
// descriptor built (`llvm.insertvalue` of its allocated or aligned pointer:
// `isHeapPointer`; a memref converted to one) views a heap payload, and none
// comes from outside (no public function takes, no external function other
// than cleave's runtime returns, no LLVM global holds, a pointer or a
// descriptor). Then a descriptor anywhere (a field loaded, extracted,
// carried by a loop, passed to a private function) is one of those.
static bool descriptorsAreHeapPayloads(ModuleOp module, HeapMemo &memo) {
  auto key = std::make_pair(module.getOperation(), ~0u);
  if (auto known = memo.find(key); known != memo.end())
    return known->second;
  memo[key] = false; // while deciding: the conservative answer
  bool heap = true;
  module.walk([&](Operation *op) {
    if (!heap)
      return WalkResult::interrupt();
    if (auto insert = dyn_cast<LLVM::InsertValueOp>(op)) {
      auto [container, index] = lastContainer(insert.getContainer().getType(), insert.getPosition());
      if (container && isMemrefDescriptor(container) && (index == 0 || index == 1) &&
          !isHeapPointer(insert.getValue(), module, memo))
        heap = false;
    } else if (auto castOp = dyn_cast<UnrealizedConversionCastOp>(op)) {
      for (auto [in, out] : llvm::zip(castOp.getInputs(), castOp.getOutputs()))
        if (isa<MemRefType>(in.getType()) && holdsPointers(out.getType()) &&
            !isHeapPayload(in, module, memo))
          heap = false;
    } else if (auto func = dyn_cast<func::FuncOp>(op)) {
      if (func.isExternal()) {
        if (!func.getName().starts_with("cleave_") &&
            llvm::any_of(func.getFunctionType().getResults(), holdsPointers))
          heap = false;
      } else if (func.isPublic() && llvm::any_of(func.getFunctionType().getInputs(), holdsPointers)) {
        heap = false;
      }
    } else if (auto func = dyn_cast<LLVM::LLVMFuncOp>(op)) {
      if (func.isExternal() && !func.getName().starts_with("cleave_") &&
          holdsPointers(func.getFunctionType().getReturnType()))
        heap = false;
    } else if (auto global = dyn_cast<LLVM::GlobalOp>(op)) {
      if (holdsPointers(global.getGlobalType()))
        heap = false;
    }
    return WalkResult::advance();
  });
  memo[key] = heap;
  return heap;
}

static bool isHeapPayload(Value v, ModuleOp module, HeapMemo &memo) {
  while (Operation *def = v.getDefiningOp()) {
    if (isa<memref::AllocOp>(def))
      return true;
    if (auto view = dyn_cast<ViewLikeOpInterface>(def))
      v = view.getViewSource();
    else if (auto castOp = dyn_cast<memref::CastOp>(def))
      v = castOp.getSource();
    else
      break;
  }
  // A memref read back out of a descriptor (a struct field's tensor).
  if (auto castOp = v.getDefiningOp<UnrealizedConversionCastOp>())
    return castOp.getNumOperands() == 1 && isMemrefDescriptor(castOp.getOperand(0).getType()) &&
           descriptorsAreHeapPayloads(module, memo);
  if (v.getDefiningOp() && !isa<scf::ForOp, scf::WhileOp, scf::IfOp>(v.getDefiningOp()))
    return false;
  auto sources = heapSources(v, module);
  if (!sources)
    return false;
  auto &[key, values] = *sources;
  if (auto known = memo.find(key); known != memo.end())
    return known->second;
  memo[key] = true;
  bool heap = allHeapPayloads(values, module, memo);
  memo[key] = heap;
  return heap;
}

// Whether the address `extract` takes never reaches the memory's contents:
// it only becomes a memref descriptor's allocated or aligned pointer
// (`index_cast`, `inttoptr`, then `llvm.insertvalue` into a descriptor's
// field 0 or 1), or is handed to the runtime's refcount operations, which
// touch the header before a payload, never the payload (a free they cause
// is reported by `cleave-rt`'s free hook).
static bool onlyBuildsDescriptors(memref::ExtractAlignedPointerAsIndexOp extract) {
  auto refcounting = [](StringRef callee) {
    return callee == "cleave_retain" || callee == "cleave_release" || callee == "cleave_release_void";
  };
  SmallVector<Value> pending{extract.getResult()};
  while (!pending.empty()) {
    Value v = pending.pop_back_val();
    for (OpOperand &use : v.getUses()) {
      Operation *user = use.getOwner();
      if (isa<arith::IndexCastOp, LLVM::IntToPtrOp>(user)) {
        pending.push_back(user->getResult(0));
        continue;
      }
      if (auto call = dyn_cast<func::CallOp>(user); call && refcounting(call.getCallee()))
        continue;
      if (auto call = dyn_cast<LLVM::CallOp>(user); call && call.getCallee() && refcounting(*call.getCallee()))
        continue;
      auto insert = dyn_cast<LLVM::InsertValueOp>(user);
      if (!insert || use.getOperandNumber() != 1)
        return false; // stored elsewhere, compared, passed: escapes
      auto [container, index] = lastContainer(insert.getContainer().getType(), insert.getPosition());
      if (!container || !isMemrefDescriptor(container) || (index != 0 && index != 1))
        return false;
    }
  }
  return true;
}

// Vulkan backend: the host's accesses to memory a kernel may also have,
// reported to the runtime, which keeps device mirrors of heap payloads
// (`cleave-vk`, `cleave_vk_dispatch`):
//
//   cleave_vk_sync_host(ptr, bytes)    before a read or a write: what a
//                                      kernel wrote there comes back
//   cleave_vk_host_wrote(ptr, bytes)   after a write: the mirror is stale
//
// (`memrefSpan` gives the range.) An access in a loop or a branch is
// reported around the outermost enclosing one that launches nothing
// (no `cleave_vk_dispatch`, no call to a function of the module that may
// launch one) and that its memref is defined outside of: once per loop
// rather than once per element; still before any launch that could read,
// after any that could write. Conservative: a write in an untaken branch
// still marks the mirror stale, which only costs an upload. A call to a
// function of the module reports nothing (the callee reports its own
// accesses); a call to anything else, or an operation that doesn't declare
// its memory effects, reads and writes every memref it's given. Returns how
// many accesses it reported.
static int64_t cleaveSyncHostAccesses(MlirOperation op) {
  auto module = dyn_cast<ModuleOp>(unwrap(op));
  if (!module)
    return 0;
  MLIRContext *context = module.getContext();
  StringRef dispatchName = "cleave_vk_dispatch";
  if (!module.lookupSymbol<func::FuncOp>(dispatchName))
    return 0; // nothing launches a kernel: nothing to keep coherent

  // Functions that may launch a kernel, transitively.
  llvm::DenseSet<Operation *> launching;
  bool grew = true;
  while (grew) {
    grew = false;
    module.walk([&](func::FuncOp func) {
      if (func.isExternal() || launching.contains(func))
        return;
      bool launches = false;
      func.walk([&](func::CallOp call) {
        auto callee = module.lookupSymbol<func::FuncOp>(call.getCallee());
        if (call.getCallee() == dispatchName || (callee && launching.contains(callee)))
          launches = true;
      });
      if (launches) {
        launching.insert(func);
        grew = true;
      }
    });
  }
  auto mayLaunch = [&](Operation *region) {
    bool found = false;
    region->walk([&](Operation *inner) {
      if (auto call = dyn_cast<func::CallOp>(inner)) {
        auto callee = module.lookupSymbol<func::FuncOp>(call.getCallee());
        if (call.getCallee() == dispatchName || (callee && launching.contains(callee)))
          found = true;
      } else if (isa<func::CallIndirectOp>(inner)) {
        found = true;
      }
    });
    return found;
  };

  // An address only stored as a descriptor's pointer (a tensor put into a
  // struct field) doesn't reach memory: what later reads or writes through
  // the descriptor is a memref access again, reported where it happens,
  // provided every descriptor stays in the module (`descriptorsAreHeapPayloads`).
  HeapMemo heapMemo;
  bool descriptors = descriptorsAreHeapPayloads(module, heapMemo);
  enum Access { Read = 1, Write = 2 };
  // (where to report, memref) -> accesses
  llvm::MapVector<std::pair<Operation *, Value>, int> reports;
  auto record = [&](Operation *at, Value v, int access) {
    if (!isa<MemRefType>(v.getType()))
      return; // an unranked memref: not lowered by this backend
    Operation *point = at;
    while (true) {
      Operation *parent = point->getParentOp();
      if (!parent || isa<func::FuncOp>(parent) || parent->getNumRegions() == 0)
        break;
      if (mayLaunch(parent) || parent->isAncestor(v.getParentBlock()->getParentOp()) ||
          (v.getDefiningOp() && parent->isProperAncestor(v.getDefiningOp())))
        break;
      point = parent;
    }
    reports[{point, v}] |= access;
  };

  module.walk([&](func::FuncOp func) {
    if (func.isExternal())
      return;
    func.walk([&](Operation *inner) {
      // A memref's address taken for anything but the runtime's own calls:
      // the memory escapes to code that may read and write it (an extern
      // given a raw pointer, `sgemm`).
      if (auto extract = dyn_cast<memref::ExtractAlignedPointerAsIndexOp>(inner)) {
        if (!extract->hasAttr(kRuntimeAddress) && !(onlyBuildsDescriptors(extract) && descriptors))
          record(extract, extract.getSource(), Read | Write);
        return;
      }
      if (isa<ViewLikeOpInterface, memref::CastOp, memref::ExtractStridedMetadataOp, memref::DimOp,
              memref::DeallocOp, memref::AllocOp, memref::AllocaOp, func::ReturnOp>(inner))
        return;
      if (auto call = dyn_cast<func::CallOp>(inner)) {
        StringRef callee = call.getCallee();
        auto target = module.lookupSymbol<func::FuncOp>(callee);
        if (callee.starts_with("cleave_vk_") || (target && !target.isExternal()))
          return;
        for (Value operand : call.getOperands())
          if (isa<BaseMemRefType>(operand.getType()))
            record(call, operand, Read | Write);
        return;
      }
      if (inner->hasTrait<OpTrait::HasRecursiveMemoryEffects>() || inner->getNumRegions() != 0)
        return;
      auto effects = dyn_cast<MemoryEffectOpInterface>(inner);
      if (!effects) {
        for (Value operand : inner->getOperands())
          if (isa<BaseMemRefType>(operand.getType()))
            record(inner, operand, Read | Write);
        return;
      }
      SmallVector<MemoryEffects::EffectInstance> instances;
      effects.getEffects(instances);
      for (auto &effect : instances) {
        Value v = effect.getValue();
        if (!v || !isa<BaseMemRefType>(v.getType()))
          continue;
        if (isa<MemoryEffects::Read>(effect.getEffect()))
          record(inner, v, Read);
        else if (isa<MemoryEffects::Write>(effect.getEffect()))
          record(inner, v, Write);
      }
    });
  });
  if (reports.empty())
    return 0;

  OpBuilder builder(context);
  Type i64 = builder.getI64Type();
  Location unknown = builder.getUnknownLoc();
  for (StringRef name : {"cleave_vk_sync_host", "cleave_vk_host_wrote"})
    if (!module.lookupSymbol<func::FuncOp>(name)) {
      builder.setInsertionPointToStart(module.getBody());
      func::FuncOp::create(builder, unknown, name, builder.getFunctionType({i64, i64}, {})).setPrivate();
    }
  for (auto &[key, access] : reports) {
    auto [point, v] = key;
    Location loc = point->getLoc();
    builder.setInsertionPoint(point);
    auto [first, bytes] = memrefSpan(builder, loc, v);
    func::CallOp::create(builder, loc, "cleave_vk_sync_host", TypeRange{}, ValueRange{first, bytes});
    if (access & Write) {
      builder.setInsertionPointAfter(point);
      func::CallOp::create(builder, loc, "cleave_vk_host_wrote", TypeRange{}, ValueRange{first, bytes});
    }
  }
  return reports.size();
}

// Whether `call` is the first use of the buffer `given`: nothing reaches it
// (through views, at any depth) before the call, and the call doesn't repeat
// over it (in a loop the buffer's block holds).
static bool firstUseIsCall(Value given, func::CallOp call) {
  Block *block = given.getParentBlock();
  Operation *callHolder = block->findAncestorOpInBlock(*call);
  if (!callHolder)
    return false;
  for (Operation *p = call->getParentOp(); p && p != callHolder->getParentOp(); p = p->getParentOp())
    if (isa<LoopLikeOpInterface>(p))
      return false;
  SmallVector<Value> pending{given};
  while (!pending.empty()) {
    Value v = pending.pop_back_val();
    for (Operation *user : v.getUsers()) {
      if (user == call)
        continue;
      if (isa<ViewLikeOpInterface, memref::CastOp>(user)) {
        for (Value result : user->getResults())
          pending.push_back(result);
        continue;
      }
      Operation *holder = block->findAncestorOpInBlock(*user);
      if (!holder || holder == callHolder || holder->isBeforeInBlock(callHolder))
        return false;
    }
  }
  return true;
}

// Whether `v` is a whole buffer nothing else in its function can reach, its
// contents undefined: a `memref.alloc`, or a private function's parameter
// that every call site gives such a buffer, the call its first use
// (`firstUseIsCall`), passed to no other parameter (an out-parameter
// `buffer-results-to-out-params` allocated for the call). `inProgress`: a
// recursion through it, answered no.
static bool isFreshUnaliased(Value v, ModuleOp module, llvm::DenseMap<Value, bool> &memo) {
  if (v.getDefiningOp<memref::AllocOp>())
    return true;
  auto arg = dyn_cast<BlockArgument>(v);
  auto func = arg ? dyn_cast<func::FuncOp>(arg.getOwner()->getParentOp()) : func::FuncOp();
  if (!func || arg.getOwner() != &func.getBody().front() || func.isPublic())
    return false;
  if (auto known = memo.find(v); known != memo.end())
    return known->second;
  memo[v] = false;
  auto uses = SymbolTable::getSymbolUses(func, module);
  bool fresh = uses.has_value();
  for (const SymbolTable::SymbolUse &use : uses ? *uses : SymbolTable::UseRange({})) {
    auto call = dyn_cast<func::CallOp>(use.getUser());
    if (!call) {
      if (const char *d = std::getenv("CLEAVE_DEBUG_PARALLELIZE"); d && std::string(d) == "1")
        llvm::errs() << "  not fresh: " << func.getName() << " used by " << use.getUser()->getName() << '\n';
      fresh = false;
      break;
    }
    Value given = call.getOperand(arg.getArgNumber());
    bool sharedWithAnother = llvm::any_of(llvm::enumerate(call.getOperands()), [&](auto other) {
      return other.index() != arg.getArgNumber() && isa<BaseMemRefType>(other.value().getType()) &&
             viewRoot(other.value()) == viewRoot(given);
    });
    bool usedBefore = !firstUseIsCall(given, call);
    if (viewRoot(given) != given || sharedWithAnother || usedBefore || !isFreshUnaliased(given, module, memo)) {
      if (const char *d = std::getenv("CLEAVE_DEBUG_PARALLELIZE"); d && std::string(d) == "1")
        llvm::errs() << "  not fresh: " << func.getName() << " #" << arg.getArgNumber() << " at a call in "
                     << call->getParentOfType<func::FuncOp>().getName() << ": view " << (viewRoot(given) != given)
                     << " shared " << sharedWithAnother << " used before " << usedBefore << '\n';
      fresh = false;
      break;
    }
  }
  memo[v] = fresh;
  return fresh;
}

// Vulkan backend: a call to `cleave_vk_discard(ptr, bytes)` before every
// `linalg` op that overwrites the whole of a fresh allocation without
// reading it (an output its payload doesn't use, indexed by a permutation of
// the loops, no input reading the same allocation; `isFreshUnaliased`, so an
// out-parameter too): the runtime then
// doesn't upload what the host left there (a fresh allocation's garbage, a
// previous step's values). And one before the first use of every
// `memref.alloc` whose first use is (or holds) a kernel: what a fresh
// allocation holds is undefined until something writes it, so there's
// nothing to upload whatever that kernel reads (`causal_softmax`'s weights,
// written by a loop the device runs, read back by it to normalize them).
// The same for an out-parameter every caller allocates fresh for the call
// (`isFreshUnaliased`) at its function's entry: `rope`'s and `rms_norm`'s
// results.
// Everything else a kernel is bound to is uploaded whenever the host wrote
// it since. Returns how many it marked.
static int64_t cleaveMarkFullOverwrites(MlirOperation op) {
  auto module = dyn_cast<ModuleOp>(unwrap(op));
  if (!module)
    return 0;
  llvm::DenseMap<Value, bool> fresh;
  SmallVector<std::pair<Operation *, Value>> overwrites;
  // `fresh` (undefined until written) in `block`: a discard before its first
  // use, if that use is or holds a kernel.
  auto discardBeforeFirstUse = [&](Value freshMemref, Block *block) {
    auto type = dyn_cast<MemRefType>(freshMemref.getType());
    if (!type || !type.hasStaticShape())
      return;
    // Every use, through views, as the operation of `block` that holds it.
    Operation *first = nullptr;
    SmallVector<Operation *> users;
    SmallVector<Value> pending{freshMemref};
    while (!pending.empty()) {
      Value v = pending.pop_back_val();
      for (Operation *user : v.getUsers()) {
        if (isa<ViewLikeOpInterface, memref::CastOp>(user))
          for (Value result : user->getResults())
            pending.push_back(result);
        users.push_back(user);
        Operation *holder = block->findAncestorOpInBlock(*user);
        if (holder && (!first || holder->isBeforeInBlock(first)))
          first = holder;
      }
    }
    if (!first || isa<ViewLikeOpInterface, memref::CastOp>(first))
      return;
    bool kernel = llvm::any_of(users, [&](Operation *user) {
      return block->findAncestorOpInBlock(*user) == first &&
             (isa<linalg::LinalgOp>(user) || user->getParentOfType<scf::ParallelOp>());
    });
    if (kernel)
      overwrites.push_back({first, freshMemref});
  };
  module.walk([&](memref::AllocOp alloc) {
    if (!alloc->getParentOfType<gpu::GPUModuleOp>())
      discardBeforeFirstUse(alloc.getResult(), alloc->getBlock());
  });
  module.walk([&](func::FuncOp func) {
    if (func.isExternal() || func->getParentOfType<gpu::GPUModuleOp>())
      return;
    for (BlockArgument arg : func.getArguments())
      if (isa<MemRefType>(arg.getType()) && isFreshUnaliased(arg, module, fresh))
        discardBeforeFirstUse(arg, &func.front());
  });
  module.walk([&](linalg::LinalgOp linalgOp) {
    if (!linalgOp.hasPureBufferSemantics() || linalgOp->getParentOfType<gpu::GPUModuleOp>())
      return;
    for (OpOperand &init : linalgOp.getDpsInitsMutable()) {
      Value out = init.get();
      auto type = dyn_cast<MemRefType>(out.getType());
      // A fresh allocation, which no input of the op reads (an in-place
      // `o * w` reads `o` through an input, though its payload ignores the
      // output's value): only then is nothing there worth uploading. A
      // memref from anywhere else (a struct field, a parameter) may alias an
      // input under another value.
      bool readByAnInput = llvm::any_of(linalgOp.getDpsInputOperands(),
                                        [&](OpOperand *input) { return viewRoot(input->get()) == viewRoot(out); });
      if (!type || !type.hasStaticShape() || linalgOp.payloadUsesValueFromOperand(&init) ||
          !linalgOp.getMatchingIndexingMap(&init).isPermutation() || viewRoot(out) != out ||
          readByAnInput || !isFreshUnaliased(out, module, fresh))
        continue;
      if (llvm::is_contained(overwrites, std::pair<Operation *, Value>(linalgOp, out)))
        continue;
      overwrites.push_back({linalgOp, out});
    }
  });
  if (overwrites.empty())
    return 0;
  OpBuilder builder(module.getContext());
  Type i64 = builder.getI64Type();
  StringRef name = "cleave_vk_discard";
  if (!module.lookupSymbol<func::FuncOp>(name)) {
    builder.setInsertionPointToStart(module.getBody());
    func::FuncOp::create(builder, builder.getUnknownLoc(), name, builder.getFunctionType({i64, i64}, {})).setPrivate();
  }
  for (auto [linalgOp, out] : overwrites) {
    builder.setInsertionPoint(linalgOp);
    auto [first, bytes] = memrefSpan(builder, linalgOp->getLoc(), out);
    func::CallOp::create(builder, linalgOp->getLoc(), name, TypeRange{}, ValueRange{first, bytes});
  }
  return overwrites.size();
}

// Vulkan backend: every `scf.for` of a kernel that yields values, run only
// when it has an iteration (`scf.if (lb < ub)`), its initial values the
// results otherwise. MLIR's SCF-to-SPIR-V conversion carries a loop's results
// in variables the loop body stores at each `scf.yield`: a loop that doesn't
// run never stores them, and its results read uninitialized memory
// (`causal_softmax`'s first row, `for j in 1..i + 1` from `m = s[i, 0]`: a NaN,
// sometimes). An `scf.if`'s results are stored by either branch. Returns how
// many loops it guarded.
static int64_t cleaveGuardZeroTripLoops(MlirOperation op) {
  SmallVector<scf::ForOp> loops;
  unwrap(op)->walk([&](scf::ForOp loop) {
    if (loop.getNumResults() > 0 && loop->getParentOfType<gpu::GPUModuleOp>())
      loops.push_back(loop);
  });
  for (scf::ForOp loop : loops) {
    OpBuilder builder(loop);
    Location loc = loop.getLoc();
    // `scf.for`'s own comparison: signed unless marked unsigned.
    Value runs = arith::CmpIOp::create(
        builder, loc, loop.getUnsignedCmp() ? arith::CmpIPredicate::ult : arith::CmpIPredicate::slt,
        loop.getLowerBound(), loop.getUpperBound());
    auto guard = scf::IfOp::create(builder, loc, loop.getResultTypes(), runs, /*withElseRegion=*/true);
    loop->replaceAllUsesWith(guard.getResults());
    Block *then = guard.thenBlock();
    loop->moveBefore(then, then->end());
    OpBuilder thenBuilder = OpBuilder::atBlockEnd(then);
    scf::YieldOp::create(thenBuilder, loc, loop.getResults());
    OpBuilder elseBuilder = OpBuilder::atBlockEnd(guard.elseBlock());
    scf::YieldOp::create(elseBuilder, loc, loop.getInitArgs());
  }
  return loops.size();
}

// Vulkan backend: the views a kernel uses (and the index arithmetic they
// take) moved into its `gpu.launch`, for `fold-memref-alias-ops` to fold
// them into the accesses: the kernel then takes the memref they view, whose
// layout SPIR-V can express, where a view at an offset known only at run
// time (`update(x, row, r, 0)`'s row `r`) has none. Returns how many
// launches it changed.
static int64_t cleaveSinkViewsIntoLaunches(MlirOperation op) {
  SmallVector<gpu::LaunchOp> launches;
  unwrap(op)->walk([&](gpu::LaunchOp launch) { launches.push_back(launch); });
  int64_t changed = 0;
  for (gpu::LaunchOp launch : launches) {
    auto sinkable = [](Operation *candidate) {
      return isa<ViewLikeOpInterface, memref::CastOp, arith::ConstantOp, arith::IndexCastOp, arith::AddIOp,
                 arith::SubIOp, arith::MulIOp, affine::AffineApplyOp>(candidate);
    };
    if (succeeded(sinkOperationsIntoLaunchOp(launch, sinkable)))
      ++changed;
  }
  return changed;
}

// Vulkan backend: every `gpu.launch_func` becomes a call to the runtime,
//
//   cleave_vk_dispatch(spirv, words, entry, groups_x, groups_y, groups_z,
//                      count, pointers, bytes, flags)  (every argument an i64)
//
// `spirv` the binary of the kernel's `spirv.module` (`convert-gpu-to-spirv`
// names it `__spv__<gpu.module>`), serialized here and embedded as a constant
// global; `entry` its entry point's name, NUL-terminated; `pointers` and
// `bytes` and `flags` three stack arrays of `count` entries, each kernel
// argument's buffer (from the start of what it views, its offset included:
// `memrefSpan`), size in bytes, and what the runtime may do with
// it (1: a heap payload, which may stay mirrored, `isHeapPayload`; 2: the
// kernel writes it; 4: the kernel reads it), in binding order. The kernels' modules are then
// removed: what remains is host code for the CPU pipeline. Every kernel
// argument must be a statically shaped memref laid out contiguously. Returns
// how many launches it lowered.
static int64_t cleaveLowerGpuLaunches(MlirOperation op) {
  auto module = dyn_cast<ModuleOp>(unwrap(op));
  if (!module)
    return 0;
  MLIRContext *context = module.getContext();
  // A loop mapped for the GPU that `convert-parallel-loops-to-gpu` left as
  // it was runs on the host, one element at a time: correct, and slow
  // enough to say so (MNIST's fills, before their bounds were folded).
  module.walk([&](scf::ParallelOp loop) {
    if (loop->hasAttr("mapping"))
      loop.emitWarning("a parallel loop meant for the GPU stayed on the host");
  });

  SmallVector<gpu::LaunchFuncOp> launches;
  module.walk([&](gpu::LaunchFuncOp launch) { launches.push_back(launch); });
  if (launches.empty())
    return 0;

  OpBuilder builder(context);
  Type i64 = builder.getI64Type();
  Location unknown = builder.getUnknownLoc();
  StringRef dispatchName = "cleave_vk_dispatch";
  if (!module.lookupSymbol<func::FuncOp>(dispatchName)) {
    builder.setInsertionPointToStart(module.getBody());
    auto decl = func::FuncOp::create(builder, unknown, dispatchName,
                                     builder.getFunctionType(SmallVector<Type>(10, i64), {}));
    decl.setPrivate();
  }

  // One constant global per kernel module (its SPIR-V words) and per kernel
  // (its entry point's name).
  llvm::StringMap<memref::GlobalOp> globals;
  auto constantGlobal = [&](StringRef name, DenseElementsAttr value) -> memref::GlobalOp {
    if (auto found = globals.find(name); found != globals.end())
      return found->second;
    builder.setInsertionPointToStart(module.getBody());
    auto type = MemRefType::get({value.getNumElements()}, value.getElementType());
    auto global = memref::GlobalOp::create(builder, unknown, name, builder.getStringAttr("private"), type, value,
                                           /*constant=*/true, /*alignment=*/builder.getI64IntegerAttr(4));
    globals[name] = global;
    return global;
  };

  HeapMemo heapParams;
  int64_t lowered = 0;
  for (gpu::LaunchFuncOp launch : launches) {
    StringRef kernelModule = launch.getKernelModuleName().getValue();
    StringRef kernel = launch.getKernelName().getValue();
    std::string spirvName = ("__spv__" + kernelModule).str();
    spirv::ModuleOp spirvModule;
    module.walk([&](spirv::ModuleOp candidate) {
      if (candidate.getName() == spirvName)
        spirvModule = candidate;
    });
    if (!spirvModule) {
      launch.emitError("no SPIR-V module for this kernel (`convert-gpu-to-spirv` didn't run?)");
      continue;
    }
    SmallVector<uint32_t> words;
    if (failed(spirv::serialize(spirvModule, words))) {
      launch.emitError("the kernel's SPIR-V module doesn't serialize");
      continue;
    }
    SmallVector<APInt> wordValues;
    for (uint32_t word : words)
      wordValues.push_back(APInt(32, word));
    auto wordsType = RankedTensorType::get({(int64_t)words.size()}, builder.getI32Type());
    memref::GlobalOp binary =
        constantGlobal(("__cleave_spirv_" + kernelModule).str(), DenseElementsAttr::get(wordsType, wordValues));
    SmallVector<APInt> nameBytes;
    for (char c : kernel)
      nameBytes.push_back(APInt(8, (uint8_t)c));
    nameBytes.push_back(APInt(8, 0));
    auto nameType = RankedTensorType::get({(int64_t)nameBytes.size()}, builder.getI8Type());
    memref::GlobalOp entry = constantGlobal(("__cleave_entry_" + kernelModule + "_" + kernel).str(),
                                            DenseElementsAttr::get(nameType, nameBytes));

    builder.setInsertionPoint(launch);
    Location loc = launch.getLoc();
    auto asI64 = [&](Value v) -> Value { return arith::IndexCastOp::create(builder, loc, i64, v); };
    auto constI64 = [&](int64_t v) -> Value { return arith::ConstantIntOp::create(builder, loc, i64, v); };
    auto constIndex = [&](int64_t v) -> Value { return arith::ConstantIndexOp::create(builder, loc, v); };
    auto address = [&](Value ref) -> Value {
      auto extract = memref::ExtractAlignedPointerAsIndexOp::create(builder, loc, ref);
      extract->setAttr(kRuntimeAddress, builder.getUnitAttr());
      return asI64(extract);
    };
    auto addressOf = [&](memref::GlobalOp global) -> Value {
      return address(memref::GetGlobalOp::create(builder, loc, global.getType(), global.getSymName()));
    };

    auto operands = launch.getKernelOperands();
    int64_t count = operands.size();
    auto arrayType = MemRefType::get({count}, i64);
    Value pointers = memref::AllocaOp::create(builder, loc, arrayType);
    Value bytes = memref::AllocaOp::create(builder, loc, arrayType);
    Value flags = memref::AllocaOp::create(builder, loc, arrayType);
    // Which of the kernel's arguments it writes: a store into it, or into a
    // view of it, in the kernel.
    auto kernelFunc = SymbolTable::lookupNearestSymbolFrom<gpu::GPUFuncOp>(launch, launch.getKernel());
    // Whether the kernel reads (`reads`) or writes its argument `i`: an
    // access to it, or to a view of it; anything undeclared counts as both.
    auto accesses = [&](unsigned i, bool reads) {
      if (!kernelFunc)
        return true;
      bool found = false;
      kernelFunc.walk([&](Operation *inner) {
        auto effects = dyn_cast<MemoryEffectOpInterface>(inner);
        if (!effects) {
          if (!inner->hasTrait<OpTrait::HasRecursiveMemoryEffects>() && inner->getNumRegions() == 0 &&
              !isa<ViewLikeOpInterface>(inner) &&
              llvm::any_of(inner->getOperands(), [](Value o) { return isa<BaseMemRefType>(o.getType()); }))
            found = true;
          return;
        }
        SmallVector<MemoryEffects::EffectInstance> instances;
        effects.getEffects(instances);
        for (auto &effect : instances) {
          bool kind = reads ? isa<MemoryEffects::Read>(effect.getEffect()) : isa<MemoryEffects::Write>(effect.getEffect());
          if (kind && (!effect.getValue() || viewRoot(effect.getValue()) == kernelFunc.getArgument(i)))
            found = true;
        }
      });
      return found;
    };
    bool ok = true;
    for (auto [i, operand] : llvm::enumerate(operands)) {
      // A scalar (a value known only at run time, `Scale::scale`'s factor):
      // a one-element buffer on the stack, as `convert-gpu-to-spirv` wraps
      // it in a one-element storage buffer; an `index` as SPIR-V's 32-bit
      // one. Copied, never mirrored, only read.
      if (operand.getType().isIntOrIndexOrFloat()) {
        Value scalar = operand;
        if (scalar.getType().isIndex())
          scalar = arith::IndexCastOp::create(builder, loc, builder.getI32Type(), scalar);
        Type scalarType = scalar.getType();
        Value cell = memref::AllocaOp::create(builder, loc, MemRefType::get({1}, scalarType));
        memref::StoreOp::create(builder, loc, scalar, cell, constIndex(0));
        Value slot = constIndex(i);
        memref::StoreOp::create(builder, loc, address(cell), pointers, slot);
        memref::StoreOp::create(builder, loc, constI64(std::max(4u, scalarType.getIntOrFloatBitWidth() / 8)), bytes,
                                slot);
        memref::StoreOp::create(builder, loc, constI64(4), flags, slot);
        continue;
      }
      auto type = dyn_cast<MemRefType>(operand.getType());
      SmallVector<int64_t> strides;
      int64_t offset;
      if (!type || !type.hasStaticShape() || failed(type.getStridesAndOffset(strides, offset)) ||
          !type.getElementType().isIntOrFloat()) {
        launch.emitError("a GPU kernel argument must be a number or a statically shaped memref of numbers");
        ok = false;
        break;
      }
      // Contiguous, row-major: each stride the product of the dimensions after it.
      int64_t expected = 1;
      for (int64_t d = type.getRank() - 1; d >= 0; --d) {
        if (strides[d] != expected) {
          launch.emitError("a GPU kernel argument must be laid out contiguously");
          ok = false;
          break;
        }
        expected *= type.getDimSize(d);
      }
      if (!ok)
        break;
      // Bound from the start of the buffer it views (`memrefSpan`).
      auto [first, span] = memrefSpan(builder, loc, operand, /*fromBase=*/true);
      Value slot = constIndex(i);
      memref::StoreOp::create(builder, loc, first, pointers, slot);
      memref::StoreOp::create(builder, loc, span, bytes, slot);
      bool read = accesses(i, true);
      int64_t flag = (isHeapPayload(operand, module, heapParams) ? 1 : 0) | (accesses(i, false) ? 2 : 0) |
                     (read ? 4 : 0);
      memref::StoreOp::create(builder, loc, constI64(flag), flags, slot);
    }
    if (!ok)
      continue;
    gpu::KernelDim3 grid = launch.getGridSizeOperandValues();
    Value pointersAddress = address(pointers);
    Value bytesAddress = address(bytes);
    Value flagsAddress = address(flags);
    func::CallOp::create(builder, loc, dispatchName, TypeRange{},
                         ValueRange{addressOf(binary), constI64(words.size()), addressOf(entry), asI64(grid.x),
                                    asI64(grid.y), asI64(grid.z), constI64(count), pointersAddress, bytesAddress,
                                    flagsAddress});
    launch.erase();
    ++lowered;
  }

  // The kernels now live in the globals: their modules go.
  SmallVector<Operation *> kernelModules;
  for (Operation &child : module.getBody()->getOperations())
    if (isa<gpu::GPUModuleOp, spirv::ModuleOp>(child))
      kernelModules.push_back(&child);
  for (Operation *kernelModule : kernelModules)
    kernelModule->erase();
  module->removeAttr(gpu::GPUDialect::getContainerModuleAttrName());
  return lowered;
}

static int64_t cleaveForwardReadOnlyArguments(MlirOperation op) {
  auto module = dyn_cast<ModuleOp>(unwrap(op));
  if (!module)
    return 0;
  ReadOnlyArguments memo;
  SmallVector<memref::CopyOp> copies;
  module->walk([&](memref::CopyOp copy) { copies.push_back(copy); });
  int64_t forwarded = 0;
  for (memref::CopyOp copy : copies) {
    auto cast = copy.getSource().getDefiningOp<memref::CastOp>();
    auto alloc = copy.getTarget().getDefiningOp<memref::AllocOp>();
    if (!cast || !alloc || cast.getSource().getType() != alloc.getType() || alloc->getBlock() != copy->getBlock() ||
        !alloc->isBeforeInBlock(copy))
      continue;
    Value buffer = cast.getSource();
    Block *block = copy->getBlock();
    // The copy's only uses: the copy, calls after it in this block reading
    // it, and its frees.
    SmallVector<OpOperand *> calls;
    SmallVector<Operation *> frees;
    Operation *last = copy.getOperation();
    bool ok = true;
    for (OpOperand &use : alloc.getResult().getUses()) {
      Operation *user = use.getOwner();
      if (user == copy.getOperation())
        continue;
      if (isa<memref::DeallocOp>(user)) {
        frees.push_back(user);
        continue;
      }
      if (!isa<func::CallOp>(user) || user->getBlock() != block || !copy->isBeforeInBlock(user) ||
          !useOnlyReads(use, module, memo)) {
        ok = false;
        break;
      }
      calls.push_back(&use);
      if (last->isBeforeInBlock(user))
        last = user;
    }
    if (!ok || calls.empty())
      continue;
    // Nothing writes `buffer` (or a view of it) between the copy and the
    // last call.
    Value root = viewRoot(buffer);
    for (Operation *o = copy->getNextNode(); ok && o && o != last->getNextNode(); o = o->getNextNode())
      o->walk([&](Operation *inner) {
        for (OpOperand &operand : inner->getOpOperands())
          if (isa<BaseMemRefType>(operand.get().getType()) && viewRoot(operand.get()) == root &&
              !useOnlyReads(operand, module, memo))
            ok = false;
      });
    if (!ok)
      continue;
    for (OpOperand *use : calls)
      use->set(buffer);
    copy.erase();
    for (Operation *free : frees)
      free->erase();
    alloc.erase();
    if (cast->use_empty())
      cast.erase();
    ++forwarded;
  }
  return forwarded;
}

// A `scf.while` whose condition is false on entry: its "before" region runs
// once and the loop ends, so the loop is that region inlined, its results
// the condition's forwarded values. The condition is decided by folding the
// region's operations on the loop's constant initial values (a `for i in
// 0..0`), so a body that never runs isn't lowered at all, whatever the
// optimization level, and nothing it calls has to exist. Returns how many
// loops it removed.
static int64_t cleaveDropZeroTripWhiles(MlirOperation op) {
  SmallVector<scf::WhileOp> loops;
  unwrap(op)->walk([&](scf::WhileOp loop) { loops.push_back(loop); });
  int64_t dropped = 0;
  for (scf::WhileOp loop : loops) {
    Block &before = loop.getBefore().front();
    llvm::DenseMap<Value, Attribute> known;
    auto constant = [&](Value v) -> Attribute {
      if (auto found = known.find(v); found != known.end())
        return found->second;
      Attribute attr;
      return matchPattern(v, m_Constant(&attr)) ? attr : Attribute();
    };
    for (auto [arg, init] : llvm::zip(before.getArguments(), loop.getInits()))
      if (Attribute attr = constant(init))
        known[arg] = attr;
    for (Operation &inner : before.without_terminator()) {
      if (inner.getNumRegions() != 0 || !isMemoryEffectFree(&inner))
        continue;
      // Only on known operands: a fold may otherwise rewrite the operation
      // in place (`arith.cmpi` moving a constant to the right).
      SmallVector<Attribute> operands;
      for (Value operand : inner.getOperands())
        operands.push_back(constant(operand));
      if (llvm::is_contained(operands, Attribute()))
        continue;
      SmallVector<OpFoldResult> results;
      if (failed(inner.fold(operands, results)) || results.size() != inner.getNumResults())
        continue;
      for (auto [result, folded] : llvm::zip(inner.getResults(), results)) {
        Attribute attr = isa<Attribute>(folded) ? cast<Attribute>(folded) : constant(cast<Value>(folded));
        if (attr)
          known[result] = attr;
      }
    }
    auto condition = cast<scf::ConditionOp>(before.getTerminator());
    auto decided = dyn_cast_or_null<IntegerAttr>(constant(condition.getCondition()));
    if (!decided || !decided.getValue().isZero())
      continue;
    IRRewriter rewriter(loop.getContext());
    SmallVector<Value> inits(loop.getInits());
    rewriter.inlineBlockBefore(&before, loop, inits);
    rewriter.replaceOp(loop, condition.getArgs());
    rewriter.eraseOp(condition);
    ++dropped;
  }
  return dropped;
}

// The only consumer of `product`'s result when it is pointwise over the
// product's rows (two parallel loops, one result written through the
// identity, every operand read through a projected permutation, the product
// through the identity); null otherwise.
static linalg::GenericOp pointwiseConsumer(linalg::MatmulOp product) {
  Value result = product->getResult(0);
  if (!result.hasOneUse())
    return nullptr;
  auto consumer = dyn_cast<linalg::GenericOp>(*result.getUsers().begin());
  if (!consumer || !consumer.hasPureTensorSemantics() || consumer.getNumLoops() != 2 ||
      consumer.getNumParallelLoops() != 2 || consumer.getNumDpsInits() != 1)
    return nullptr;
  if (!llvm::all_of(consumer.getIndexingMapsArray(), [](AffineMap m) { return m.isProjectedPermutation(); }) ||
      !consumer.getMatchingIndexingMap(consumer.getDpsInitOperand(0)).isIdentity())
    return nullptr;
  for (OpOperand &input : consumer->getOpOperands())
    if (input.get() == result && !consumer.getMatchingIndexingMap(&input).isIdentity())
      return nullptr;
  return consumer;
}

// A product BLAS computes (`linalg.matmul` marked `cleave.blas`,
// `stdlib/linalg/matrix.cleave` above `BLAS_MIN_WORK`) whose only consumer
// is an elementwise op (a bias, an activation, a residual, a gradient's
// pointwise factor): the consumer is tiled by `rows` rows and the product,
// with what initializes it (its zero fill, a broadcast bias), fused into the
// same loop. Each tile of the product is then consumed while it is still in
// L2 instead of written to memory whole and read back: with every core busy,
// memory gives a core ~6 GB/s where its L2 gives ~220. Before
// `cleaveLowerBlasMatmuls`, on tensors. Returns how many products it fused.
static int64_t cleaveBlasTileAndFuse(MlirOperation op, int64_t rows) {
  // A consumer of several products (SwiGLU's `silu(gate) * up`) once: all
  // of them are fused into its loop.
  llvm::SetVector<linalg::GenericOp> consumers;
  unwrap(op)->walk([&](linalg::MatmulOp product) {
    // Only what `cleaveLowerBlasMatmuls` will make a `sgemm` (`f32`).
    if (!product->hasAttr("cleave.blas") || !product.hasPureTensorSemantics() ||
        !getElementTypeOrSelf(product->getResult(0).getType()).isF32())
      return;
    if (linalg::GenericOp consumer = pointwiseConsumer(product))
      consumers.insert(consumer);
  });
  if (consumers.empty())
    return 0;
  MLIRContext *context = unwrap(op)->getContext();
  IRRewriter rewriter(context);
  int64_t fused = 0;
  for (linalg::GenericOp consumer : consumers) {
    scf::SCFTilingOptions tiling;
    tiling.setTileSizes(getAsIndexOpFoldResult(context, ArrayRef<int64_t>{rows, 0}));
    tiling.setLoopType(scf::SCFTilingOptions::LoopType::ForOp);
    scf::SCFTileAndFuseOptions options;
    options.setTilingOptions(tiling);
    // What is fused: the products and what initializes them (a fill, a
    // broadcast bias: any `linalg` op with only parallel loops, cheap to
    // compute per tile), each only if every use of it is the consumer or
    // another op fused with it. A producer with any other use would be
    // computed twice, per tile here and whole for the others (a product the
    // backward pass reads again: a `sgemm` too many).
    llvm::SmallPtrSet<Operation *, 8> chain;
    {
      auto fusable = [](Operation *o) {
        auto linalgOp = dyn_cast<linalg::LinalgOp>(o);
        if (auto product = dyn_cast<linalg::MatmulOp>(o))
          return product->hasAttr("cleave.blas") && product.hasPureTensorSemantics();
        return linalgOp && linalgOp.hasPureTensorSemantics() && linalgOp.getNumLoops() == linalgOp.getNumParallelLoops();
      };
      SmallVector<Operation *> worklist{consumer.getOperation()};
      llvm::SmallPtrSet<Operation *, 8> seen{consumer.getOperation()};
      while (!worklist.empty()) {
        Operation *user = worklist.pop_back_val();
        for (Value operand : user->getOperands()) {
          Operation *producer = operand.getDefiningOp();
          if (!producer || !fusable(producer) || !seen.insert(producer).second)
            continue;
          bool onlyInChain = llvm::all_of(producer->getUsers(), [&](Operation *u) {
            return u == consumer.getOperation() || chain.contains(u) || u == user;
          });
          if (!onlyInChain)
            continue;
          chain.insert(producer);
          worklist.push_back(producer);
        }
      }
    }
    options.setFusionControlFn(
        [&chain](tensor::ExtractSliceOp, OpResult producer,
                 bool) -> std::optional<scf::SCFTileAndFuseOptions::ControlFnResult> {
          if (chain.contains(producer.getOwner()))
            return scf::SCFTileAndFuseOptions::ControlFnResult{};
          return std::nullopt;
        });
    rewriter.setInsertionPoint(consumer);
    FailureOr<scf::SCFTileAndFuseResult> tiled =
        scf::tileConsumerAndFuseProducersUsingSCF(rewriter, cast<TilingInterface>(consumer.getOperation()), options);
    if (failed(tiled))
      continue;
    for (auto [original, replacement] : tiled->replacements)
      rewriter.replaceAllUsesWith(original, replacement);
    // The tiled consumer, vectorized here in rows of 16 columns, as the
    // `linalg` schedule does its epilogues: it writes rows of a larger
    // result (a subview, after bufferization), which `--affine-super-
    // vectorize` would leave scalar, and its tile loop's induction variable
    // isn't an affine symbol `--affine-fold-memref-alias-ops` could fold the
    // subview's offset into. A vector transfer reads and writes a strided
    // view as well as a whole buffer.
    for (Operation *tiledOp : tiled->tiledAndFusedOps) {
      auto generic = dyn_cast<linalg::GenericOp>(tiledOp);
      if (!generic || generic->getNumResults() != 1 ||
          !llvm::any_of(generic->getUsers(), [](Operation *u) { return isa<tensor::InsertSliceOp>(u); }))
        continue;
      scf::SCFTilingOptions rowsOf16;
      rowsOf16.setTileSizes(getAsIndexOpFoldResult(context, ArrayRef<int64_t>{1, 16}));
      rewriter.setInsertionPoint(generic);
      FailureOr<scf::SCFTilingResult> inner =
          scf::tileUsingSCF(rewriter, cast<TilingInterface>(generic.getOperation()), rowsOf16);
      if (failed(inner) || inner->tiledOps.size() != 1)
        continue;
      rewriter.replaceOp(generic, inner->replacements);
      Operation *row = inner->tiledOps.front();
      rewriter.setInsertionPoint(row);
      FailureOr<linalg::VectorizationResult> vectorized = linalg::vectorize(rewriter, row, {1, 16}, {false, false});
      if (succeeded(vectorized))
        rewriter.replaceOp(row, vectorized->replacements);
    }
    // The untiled originals, now unused: erased here, the product before it
    // becomes a `sgemm` call no dead-code elimination would remove. Tiling
    // leaves dead slices of them in the loop first.
    for (LoopLikeOpInterface loop : tiled->loops) {
      SmallVector<Operation *> trivially;
      loop->walk([&](Operation *o) {
        if (isOpTriviallyDead(o))
          trivially.push_back(o);
      });
      for (Operation *o : trivially)
        rewriter.eraseOp(o);
    }
    // A tile of the product initialized from a slice of a whole-size empty
    // tensor gets an empty tensor of the tile's size instead: a scratch tile
    // reused from one iteration to the next, not a slice of a whole-size
    // buffer written back to memory in the end.
    SmallVector<Operation *> emptySlices;
    for (LoopLikeOpInterface loop : tiled->loops)
      loop->walk([&](tensor::ExtractSliceOp slice) {
        if (slice.getSource().getDefiningOp<tensor::EmptyOp>())
          emptySlices.push_back(slice);
      });
    if (!emptySlices.empty()) {
      RewritePatternSet patterns(context);
      tensor::populateFoldTensorEmptyPatterns(patterns);
      GreedyRewriteConfig config;
      config.setStrictness(GreedyRewriteStrictness::ExistingAndNewOps);
      config.enableFolding(false);
      (void)applyOpPatternsGreedily(emptySlices, std::move(patterns), config);
    }
    SmallVector<Operation *> dead{consumer.getOperation()};
    while (!dead.empty()) {
      Operation *o = dead.pop_back_val();
      if (!o->use_empty() || !isa<linalg::LinalgOp>(o))
        continue;
      SmallVector<Operation *> producers;
      for (Value operand : o->getOperands())
        if (Operation *def = operand.getDefiningOp())
          producers.push_back(def);
      rewriter.eraseOp(o);
      dead.append(producers.begin(), producers.end());
    }
    ++fused;
  }
  return fused;
}

// A product whose row count (static) isn't a multiple of `rows`, the matmul
// schedule's row tile (`matmul_vectorize.transform.mlir`, `tile_using_forall`):
// split in two, the whole tiles and the remainder, each of static size. The
// schedule peels its inner loops (columns, `K`), but a `scf.forall` can't be
// peeled: its last tile had a dynamic size, the same operation for every
// tile, so nothing in the loop vectorized and the affine lowering rejected
// its bounds. What is split is what the schedule tiles by rows: the
// product's pointwise consumer when it is its only one (the schedule fuses
// the product into that consumer's loop, so each part here computes its own
// rows of the product), the product itself otherwise. Before the schedule, on
// tensors. Returns how many operations it split.
static int64_t cleaveSplitRowRemainders(MlirOperation op, int64_t rows) {
  llvm::SetVector<Operation *> roots;
  llvm::SmallPtrSet<Operation *, 8> fusedProducts;
  unwrap(op)->walk([&](linalg::MatmulOp product) {
    auto type = dyn_cast<RankedTensorType>(product->getResult(0).getType());
    if (!product.hasPureTensorSemantics() || !type || type.getRank() != 2 || type.isDynamicDim(0) ||
        type.getDimSize(0) <= rows || type.getDimSize(0) % rows == 0)
      return;
    if (linalg::GenericOp consumer = pointwiseConsumer(product)) {
      roots.insert(consumer.getOperation());
      fusedProducts.insert(product.getOperation());
    } else {
      roots.insert(product.getOperation());
    }
  });
  IRRewriter rewriter(unwrap(op)->getContext());
  int64_t split = 0;
  for (Operation *rootOp : roots) {
    auto root = cast<TilingInterface>(rootOp);
    if (rootOp->getNumResults() != 1)
      continue;
    rewriter.setInsertionPoint(rootOp);
    SmallVector<Range> domain = root.getIterationDomain(rewriter);
    std::optional<int64_t> m = getConstantIntValue(domain[0].size);
    if (!m)
      continue;
    SmallVector<OpFoldResult> offsets, sizes;
    for (Range range : domain) {
      offsets.push_back(range.offset);
      sizes.push_back(range.size);
    }
    // Each part's result written into its rows of the root's destination.
    Value result = cast<DestinationStyleOpInterface>(rootOp).getDpsInits()[0];
    int64_t whole = *m - *m % rows;
    for (auto [offset, size] : {std::pair{int64_t(0), whole}, std::pair{whole, *m - whole}}) {
      offsets[0] = rewriter.getIndexAttr(offset);
      sizes[0] = rewriter.getIndexAttr(size);
      rewriter.setInsertionPoint(rootOp);
      FailureOr<TilingResult> part = root.getTiledImplementation(rewriter, offsets, sizes);
      if (failed(part) || part->tiledValues.size() != 1)
        return split;
      // The consumer's rows of a product it reads: those rows of the product.
      for (Operation *slice : part->generatedSlices) {
        auto rowsOf = dyn_cast<tensor::ExtractSliceOp>(slice);
        if (!rowsOf || !fusedProducts.contains(rowsOf.getSource().getDefiningOp()))
          continue;
        rewriter.setInsertionPoint(rowsOf);
        FailureOr<TilingResult> product =
            tensor::replaceExtractSliceWithTiledProducer(rewriter, rowsOf, cast<OpResult>(rowsOf.getSource()));
        if (succeeded(product))
          rewriter.replaceOp(rowsOf, product->tiledValues);
      }
      SmallVector<OpFoldResult> resultOffsets, resultSizes;
      rewriter.setInsertionPoint(rootOp);
      if (failed(root.getResultTilePosition(rewriter, 0, offsets, sizes, resultOffsets, resultSizes)))
        return split;
      SmallVector<OpFoldResult> strides(resultOffsets.size(), rewriter.getIndexAttr(1));
      result = rewriter.create<tensor::InsertSliceOp>(rootOp->getLoc(), part->tiledValues[0], result,
                                                      resultOffsets, resultSizes, strides);
    }
    // The whole product the consumer read, now unused.
    llvm::SetVector<Operation *> products;
    for (Value operand : rootOp->getOperands())
      if (Operation *producer = operand.getDefiningOp(); producer && fusedProducts.contains(producer))
        products.insert(producer);
    rewriter.replaceOp(rootOp, result);
    for (Operation *product : products)
      if (product->use_empty())
        rewriter.eraseOp(product);
    ++split;
  }
  return split;
}

// A `linalg` op on buffers whose bounds aren't affine dimensions or symbols
// (a tile's size computed inside a `scf` loop, a tile the schedule left of
// dynamic size): `scf.for` loops, slower than the affine loops the affine
// passes vectorize, but compiled where `convert-linalg-to-affine-loops` would
// build loops the verifier rejects and fail the compilation. Whether they
// would be valid is found by building them and verifying them, diagnostics
// silenced, then erasing them: the ops they're valid for are left to
// `convert-linalg-to-affine-loops`, unchanged. Returns how many ops it lowered.
static int64_t cleaveLowerNonAffineLinalg(MlirOperation op) {
  SmallVector<linalg::LinalgOp> ops;
  unwrap(op)->walk([&](linalg::LinalgOp linalgOp) {
    if (linalgOp.hasPureBufferSemantics())
      ops.push_back(linalgOp);
  });
  MLIRContext *context = unwrap(op)->getContext();
  IRRewriter rewriter(context);
  int64_t lowered = 0;
  for (linalg::LinalgOp linalgOp : ops) {
    // What the trial inserts, the loops and their bounds' constants, sits
    // between the op's previous neighbour and the op.
    Operation *previous = linalgOp->getPrevNode();
    auto eraseTrial = [&] {
      while (Operation *inserted = linalgOp->getPrevNode()) {
        if (inserted == previous)
          break;
        rewriter.eraseOp(inserted);
      }
    };
    rewriter.setInsertionPoint(linalgOp);
    FailureOr<linalg::LinalgLoops> affine = linalg::linalgOpToAffineLoops(rewriter, linalgOp);
    if (failed(affine) || affine->empty()) {
      eraseTrial();
      continue;
    }
    bool valid;
    {
      ScopedDiagnosticHandler silenced(context, [](Diagnostic &) { return success(); });
      valid = succeeded(verify(affine->front()));
    }
    eraseTrial();
    if (valid)
      continue;
    rewriter.setInsertionPoint(linalgOp);
    if (succeeded(linalg::linalgOpToLoops(rewriter, linalgOp))) {
      rewriter.eraseOp(linalgOp);
      ++lowered;
    }
  }
  return lowered;
}

// Every `linalg.matmul` marked `cleave.blas`, tiled or not: a call to
// `cleave_blas_sgemm` (`cleave-rt`, `cblas_sgemm`) on its operands' buffers,
// on tensors, before One-Shot Bufferize. Which operand is transposed comes
// from the indexing maps (`(d2, d0)` for `A`, `(d1, d2)` for `B`); each
// leading dimension is the buffer's real row stride, read at run time (a
// tile is a view of a larger matrix). The destination's layout is dynamic
// from the `to_buffer` on: `cleaveElideBlockCopies` relies on no layout fact
// of it being known at compile time. A zero `linalg.fill` initializing the
// product is dropped for `beta = 0`; anything else (a broadcast bias, `fma`'s
// `c`) is accumulated into, `beta = 1`. Returns how many it lowered.
static int64_t cleaveLowerBlasMatmuls(MlirOperation op) {
  auto module = dyn_cast<ModuleOp>(unwrap(op));
  if (!module)
    return 0;
  SmallVector<linalg::MatmulOp> products;
  module.walk([&](linalg::MatmulOp product) {
    if (product->hasAttr("cleave.blas") && product.hasPureTensorSemantics())
      products.push_back(product);
  });
  if (products.empty())
    return 0;
  MLIRContext *context = module.getContext();
  Type i32 = IntegerType::get(context, 32), i64 = IntegerType::get(context, 64);
  FloatType f32 = Float32Type::get(context);
  auto ptrTy = LLVM::LLVMPointerType::get(context);
  auto sgemm = module.lookupSymbol<func::FuncOp>("cleave_blas_sgemm");
  if (!sgemm) {
    OpBuilder b(module.getBodyRegion());
    sgemm = b.create<func::FuncOp>(
        module.getLoc(), "cleave_blas_sgemm",
        FunctionType::get(context, {i32, i32, i32, i32, i32, f32, ptrTy, i64, i32, ptrTy, i64, i32, f32, ptrTy, i64, i32},
                          {}));
    sgemm.setPrivate();
  }
  int64_t lowered = 0;
  for (linalg::MatmulOp product : products) {
    SmallVector<AffineMap> maps = product.getIndexingMapsArray();
    MLIRContext *c = context;
    auto d = [&](unsigned i) { return getAffineDimExpr(i, c); };
    auto map2 = [&](AffineExpr x, AffineExpr y) { return AffineMap::get(3, 0, {x, y}, c); };
    bool transA = maps[0] == map2(d(2), d(0)), transB = maps[1] == map2(d(1), d(2));
    if ((!transA && maps[0] != map2(d(0), d(2))) || (!transB && maps[1] != map2(d(2), d(1))) ||
        maps[2] != map2(d(0), d(1)))
      continue;
    Value a = product.getDpsInputOperand(0)->get(), bMat = product.getDpsInputOperand(1)->get();
    Value init = product.getDpsInitOperand(0)->get();
    auto aTy = cast<RankedTensorType>(a.getType()), bTy = cast<RankedTensorType>(bMat.getType());
    auto cTy = cast<RankedTensorType>(init.getType());
    if (!aTy.hasStaticShape() || !bTy.hasStaticShape() || !cTy.hasStaticShape() || !aTy.getElementType().isF32())
      continue;
    float beta = 1.0f;
    Value dest = init;
    if (auto fill = init.getDefiningOp<linalg::FillOp>()) {
      auto zero = fill.getInputs()[0].getDefiningOp<arith::ConstantOp>();
      auto value = zero ? dyn_cast<FloatAttr>(zero.getValue()) : FloatAttr();
      if (value && value.getValue().isZero()) {
        beta = 0.0f;
        dest = fill.getOutputs()[0];
      }
    }
    OpBuilder b(product);
    Location loc = product.getLoc();
    auto dynamicLayout = StridedLayoutAttr::get(c, ShapedType::kDynamic, {ShapedType::kDynamic, ShapedType::kDynamic});
    auto bufferOf = [&](Value tensor, bool readOnly) -> Value {
      auto ty = cast<RankedTensorType>(tensor.getType());
      auto memTy = MemRefType::get(ty.getShape(), ty.getElementType(), dynamicLayout);
      auto toBuffer = b.create<bufferization::ToBufferOp>(loc, memTy, tensor);
      if (readOnly)
        toBuffer.setReadOnly(true);
      return toBuffer;
    };
    // The address of a buffer's first element and its row stride.
    auto pointerAndStride = [&](Value buffer) -> std::pair<Value, Value> {
      auto meta = b.create<memref::ExtractStridedMetadataOp>(loc, buffer);
      Value base = b.create<memref::ExtractAlignedPointerAsIndexOp>(loc, buffer);
      Value baseI64 = b.create<arith::IndexCastOp>(loc, i64, base);
      Value offset = b.create<arith::IndexCastOp>(loc, i64, meta.getOffset());
      Value four = b.create<arith::ConstantIntOp>(loc, i64, 4);
      Value address = b.create<arith::AddIOp>(loc, baseI64, b.create<arith::MulIOp>(loc, offset, four));
      Value ptr = b.create<LLVM::IntToPtrOp>(loc, ptrTy, address);
      Value stride = b.create<arith::IndexCastOp>(loc, i32, meta.getStrides()[0]);
      return {ptr, stride};
    };
    Value aBuf = bufferOf(a, true), bBuf = bufferOf(bMat, true), cBuf = bufferOf(dest, false);
    auto [aPtr, lda] = pointerAndStride(aBuf);
    auto [bPtr, ldb] = pointerAndStride(bBuf);
    auto [cPtr, ldc] = pointerAndStride(cBuf);
    int64_t m = cTy.getDimSize(0), n = cTy.getDimSize(1), k = transA ? aTy.getDimSize(0) : aTy.getDimSize(1);
    auto i32c = [&](int64_t v) -> Value { return b.create<arith::ConstantIntOp>(loc, i32, v); };
    auto i64c = [&](int64_t v) -> Value { return b.create<arith::ConstantIntOp>(loc, i64, v); };
    auto f32c = [&](float v) -> Value { return b.create<arith::ConstantFloatOp>(loc, f32, APFloat(v)); };
    b.create<func::CallOp>(loc, sgemm,
                           ValueRange{i32c(transA), i32c(transB), i32c(m), i32c(n), i32c(k), f32c(1.0f), aPtr,
                                      i64c(aTy.getNumElements()), lda, bPtr, i64c(bTy.getNumElements()), ldb,
                                      f32c(beta), cPtr, i64c(cTy.getNumElements()), ldc});
    Value result = b.create<bufferization::ToTensorOp>(loc, cTy, cBuf, /*restrict=*/true, /*writable=*/true);
    product->getResult(0).replaceAllUsesWith(result);
    Operation *fill = init.getDefiningOp<linalg::FillOp>();
    product.erase();
    if (beta == 0.0f && fill && fill->use_empty())
      fill->erase();
    ++lowered;
  }
  return lowered;
}

// An elementwise op (`linalg.generic`, every loop parallel) writing a fresh
// tensor (`tensor.empty`) while one of its operands, of the same type and
// read through the same identity map, has no other use: the op writes into
// that operand instead. Each element is read before it is written at the
// same index, so writing in place is sound, and One-Shot Bufferize still
// checks it. One buffer fewer alive at once: a smaller working set, more of
// it in cache. Only an operand computed in this function by an op that
// gives a fresh, writable result (a `linalg` op, a call): a parameter is
// borrowed, and a constant or a buffer made a tensor isn't ours to write.
// Before One-Shot Bufferize. Returns how many ops it rewrote.
static int64_t cleaveReuseDyingInputs(MlirOperation op) {
  int64_t reused = 0;
  unwrap(op)->walk([&](linalg::GenericOp generic) {
    if (!generic.hasPureTensorSemantics() || generic.getNumDpsInits() != 1 ||
        generic.getNumLoops() != generic.getNumParallelLoops())
      return;
    OpOperand *init = generic.getDpsInitOperand(0);
    if (!init->get().getDefiningOp<tensor::EmptyOp>() || !generic.getMatchingIndexingMap(init).isIdentity())
      return;
    for (OpOperand *input : generic.getDpsInputOperands()) {
      Value value = input->get();
      Operation *producer = value.getDefiningOp();
      if (!producer || !isa<linalg::LinalgOp, func::CallOp>(producer) || value.getType() != init->get().getType() ||
          !generic.getMatchingIndexingMap(input).isIdentity() || !value.hasOneUse())
        continue;
      init->set(value);
      ++reused;
      return;
    }
  });
  return reused;
}

// Removes the loop-carried values a loop yields back unchanged (`scf.for`'s
// own canonicalization, its loops alone): the loop's result is then the
// value it was given. Before `--ownership-based-buffer-deallocation`. A tile
// loop writing its tiles into a carried output buffer (`cleaveBlasTileAndFuse`)
// yields that buffer back; carried, its result is a value of unknown origin
// to the deallocation's alias analysis, which then cannot tell any buffer of
// the function from it, and frees them all together at the end of the
// function, after a run-time check of which alias which (measured on nanoLM's
// gradient: 273 buffers in one deallocation). Returns how many loop-carried
// values went.
static int64_t cleaveFoldPassthroughIterArgs(MlirOperation op) {
  SmallVector<Operation *> loops;
  int64_t before = 0;
  unwrap(op)->walk([&](scf::ForOp loop) {
    loops.push_back(loop);
    before += loop.getNumRegionIterArgs();
  });
  if (loops.empty())
    return 0;
  MLIRContext *context = unwrap(op)->getContext();
  RewritePatternSet patterns(context);
  scf::ForOp::getCanonicalizationPatterns(patterns, context);
  GreedyRewriteConfig config;
  config.setStrictness(GreedyRewriteStrictness::ExistingAndNewOps);
  config.enableFolding(false);
  (void)applyOpPatternsGreedily(loops, std::move(patterns), config);
  int64_t after = 0;
  unwrap(op)->walk([&](scf::ForOp loop) { after += loop.getNumRegionIterArgs(); });
  return before - after;
}

// Frees each buffer right after its last use. `--ownership-based-buffer-
// deallocation` frees a buffer at the end of the block its ownership ends in,
// and a function whose calls are inlined is mostly one block: every temporary
// of a synthesized gradient's forward pass stayed allocated until its return,
// those its backward pass reads and all the others (measured on nanoLM's
// gradient, `CLEAVE_ALLOC_STATS`: 9 hidden-layer-sized tensors alive per
// SwiGLU at the peak, where its backward reads 3).
//
// A deallocation is a `memref.dealloc`, alone or alone in an `scf.if` on its
// ownership flag (`--lower-deallocations`' shape). Its buffer's uses are the
// uses of everything derived from its root (`viewRoot`): views and casts,
// any memref, pointer or tensor result of an op using one (a call returning
// it, a region op yielding it), its address as an integer and integers
// computed from that. The deallocation moves to just after the last op of its
// block holding one of them, unless the buffer escapes: stored to memory, or
// used outside the block's ops. A `cleave_task_wait` marker (`spawn`, before
// `cleaveLowerSpawns`) lists the buffers its tasks use: it is a use, so a
// buffer a task reads is freed after the wait. Returns how many moved.
namespace {

bool followsAddress(Operation *op) {
  return isa<memref::ExtractAlignedPointerAsIndexOp, LLVM::PtrToIntOp, arith::IndexCastOp, arith::IndexCastUIOp,
             arith::ExtUIOp, arith::ExtSIOp, arith::TruncIOp, arith::AddIOp, arith::SubIOp, LLVM::IntToPtrOp>(op);
}

bool carriesBuffer(Type t) {
  return isa<BaseMemRefType, LLVM::LLVMPointerType, TensorType>(t);
}

// The last op of `block` using the buffer `root` (before `dealloc`), in
// `last`; false if the buffer escapes.
bool lastUseInBlock(Value root, Operation *dealloc, Operation *moved, Block *block, Operation *&last) {
  SmallVector<Value> work{root};
  llvm::SmallPtrSet<Value, 16> seen{root};
  auto follow = [&](Value v) {
    if (seen.insert(v).second)
      work.push_back(v);
  };
  while (!work.empty()) {
    Value v = work.pop_back_val();
    for (OpOperand &use : v.getUses()) {
      Operation *user = use.getOwner();
      if (user == dealloc || moved->isAncestor(user))
        continue;
      Operation *anchor = block->findAncestorOpInBlock(*user);
      if (!anchor)
        return false;
      if (auto store = dyn_cast<LLVM::StoreOp>(user); store && store.getValue() == v)
        return false;
      if (auto store = dyn_cast<memref::StoreOp>(user); store && store.getValue() == v)
        return false;
      if (!last || last->isBeforeInBlock(anchor))
        last = anchor;
      // What the use produces, and what each region op around it produces.
      for (Operation *op = user; op; op = op == anchor ? nullptr : op->getParentOp())
        for (Value r : op->getResults())
          if (carriesBuffer(r.getType()) || (r.getType().isIntOrIndex() && followsAddress(op)))
            follow(r);
    }
  }
  return true;
}

} // namespace

static int64_t cleaveDeallocAtLastUse(MlirOperation op) {
  SmallVector<memref::DeallocOp> deallocs;
  unwrap(op)->walk([&](memref::DeallocOp d) { deallocs.push_back(d); });
  int64_t moved = 0;
  for (memref::DeallocOp d : deallocs) {
    // The op to move: the dealloc, or the `scf.if` it is alone in.
    Operation *mover = d;
    Value condition;
    if (auto guard = dyn_cast<scf::IfOp>(d->getParentOp())) {
      Block *then = guard.thenBlock();
      if (guard.getNumResults() != 0 || !guard.getElseRegion().empty() || then->getOperations().size() != 2)
        continue;
      mover = guard;
      condition = guard.getCondition();
    }
    Block *block = mover->getBlock();
    Value root = viewRoot(d.getMemref());
    Operation *last = nullptr;
    if (!lastUseInBlock(root, d, mover, block, last))
      continue;
    // Where the buffer and the condition are defined, if in this block.
    for (Value v : {root, Value(d.getMemref()), condition}) {
      if (!v)
        continue;
      if (Operation *def = v.getDefiningOp())
        if (Operation *anchor = block->findAncestorOpInBlock(*def); anchor && (!last || last->isBeforeInBlock(anchor)))
          last = anchor;
    }
    if (!last || last == mover || !last->isBeforeInBlock(mover) || last->getNextNode() == mover)
      continue;
    mover->moveAfter(last);
    ++moved;
  }
  return moved;
}

// Every parallel region (`spawn`'s team, `cleaveLowerSpawns`; a parallel
// loop's, `--convert-scf-to-openmp`) starts with each member placing itself:
// `cleave_bind_worker(omp_get_thread_num())` (`cleave-rt`), one member per
// physical core. Left to the OS, two members can share one core's SMT
// siblings, and every barrier waits for that core for the whole run. Once per
// thread at run time (the call returns at once afterwards), so harmless in a
// region entered often. Returns how many regions it marked.
static int64_t cleaveBindTeams(MlirOperation op) {
  auto module = dyn_cast<ModuleOp>(unwrap(op));
  if (!module)
    return 0;
  SmallVector<omp::ParallelOp> regions;
  module.walk([&](omp::ParallelOp region) { regions.push_back(region); });
  if (regions.empty())
    return 0;
  MLIRContext *context = module.getContext();
  Type i32 = IntegerType::get(context, 32);
  auto declare = [&](StringRef name, TypeRange inputs, TypeRange results) {
    auto f = module.lookupSymbol<func::FuncOp>(name);
    if (!f) {
      OpBuilder b(module.getBodyRegion());
      f = b.create<func::FuncOp>(module.getLoc(), name, FunctionType::get(context, inputs, results));
      f.setPrivate();
    }
    return f;
  };
  auto threadNum = declare("omp_get_thread_num", {}, {i32});
  auto bind = declare("cleave_bind_worker", {i32}, {});
  for (omp::ParallelOp region : regions) {
    Block &entry = region.getRegion().front();
    OpBuilder b = OpBuilder::atBlockBegin(&entry);
    Value thread = b.create<func::CallOp>(region.getLoc(), threadNum, ValueRange{}).getResult(0);
    b.create<func::CallOp>(region.getLoc(), bind, ValueRange{thread});
  }
  return regions.size();
}

// A tensor that outlives the struct it was read from is adopted
// (`PrimOp::Adopt`, `mlir_lower.rs`): a `bufferization.clone` marked
// `cleave.adopt`, so that MLIR's ownership-based buffer deallocation, which
// runs before this, takes it for a fresh buffer it owns and releases. Each
// becomes a retain of the very same buffer instead of a copy: every buffer
// is reference counted (`cleave-unify-tensor-allocations`), so that release only drops the
// reference the retain added.
static void cleaveLowerAdoptions(MlirOperation op) {
  auto module = cast<ModuleOp>(unwrap(op));
  SmallVector<bufferization::CloneOp> clones;
  module.walk([&](bufferization::CloneOp clone) {
    if (clone->hasAttr("cleave.adopt"))
      clones.push_back(clone);
  });
  if (clones.empty())
    return;
  MLIRContext *context = module.getContext();
  auto ptrTy = LLVM::LLVMPointerType::get(context);
  auto retain = module.lookupSymbol<func::FuncOp>("cleave_retain");
  if (!retain) {
    OpBuilder b(module.getBodyRegion());
    retain = b.create<func::FuncOp>(module.getLoc(), "cleave_retain", FunctionType::get(context, {ptrTy}, {}));
    retain.setPrivate();
  }
  for (bufferization::CloneOp clone : clones) {
    OpBuilder b(clone);
    Location loc = clone.getLoc();
    Value buffer = clone.getInput();
    Value index = b.create<memref::ExtractAlignedPointerAsIndexOp>(loc, buffer);
    Value address = b.create<arith::IndexCastOp>(loc, b.getI64Type(), index);
    Value ptr = b.create<LLVM::IntToPtrOp>(loc, ptrTy, address);
    b.create<func::CallOp>(loc, retain, ValueRange{ptr});
    Value result = clone.getOutput();
    if (result.getType() != buffer.getType())
      buffer = b.create<memref::CastOp>(loc, result.getType(), buffer);
    result.replaceAllUsesWith(buffer);
    clone.erase();
  }
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
static bool approximateMath(Operation *root, ArrayRef<StringRef> approximated, bool kernelsOnly) {
  auto selected = [&](StringRef name) {
    name.consume_front("math.");
    return llvm::is_contained(approximated, name);
  };
  // Only the selected `math` ops and what their rewrites create: a module-wide
  // `applyPatternsGreedily` would also fold constants and simplify regions
  // across the whole program, work nobody asked for on 200k lines of IR.
  SmallVector<Operation *> ops;
  root->walk([&](Operation *o) {
    if (o->getDialect() && o->getDialect()->getNamespace() == "math" &&
        selected(o->getName().getStringRef()) && (!kernelsOnly || o->getParentOfType<gpu::GPUModuleOp>()))
      ops.push_back(o);
  });
  if (ops.empty())
    return true;
  RewritePatternSet patterns(root->getContext());
  populateMathPolynomialApproximationPatterns(patterns, selected);
  GreedyRewriteConfig config;
  config.setStrictness(GreedyRewriteStrictness::ExistingAndNewOps);
  config.enableFolding(false);
  return succeeded(applyOpPatternsGreedily(ops, std::move(patterns), config));
}

static bool cleaveApproximateMath(MlirOperation op) {
  static const llvm::StringRef approximated[] = {
      "tanh", "exp", "expm1", "log", "log1p", "log2", "erf", "erfc"};
  return approximateMath(unwrap(op), approximated, /*kernelsOnly=*/false);
}

// Vulkan backend: `math.sin`/`math.cos` in GPU kernels rewritten as MLIR's
// polynomial approximations (range reduction, then a polynomial, in plain
// f32 arithmetic) before `convert-math-to-spirv`, whose `GLSL.std.450`
// `Sin`/`Cos` are relaxed-precision by specification: measured on the 4070
// SUPER, 2.3e-3 relative error for `sin`, against the few ulp of the CPU's.
// What a program computes doesn't depend on the backend (`doc/plan-vulkan.md`).
// `exp`, `log`, `tanh`, `sqrt`: within an ulp or two there already.
static bool cleaveAccurateKernelMath(MlirOperation op) {
  static const llvm::StringRef approximated[] = {"sin", "cos"};
  return approximateMath(unwrap(op), approximated, /*kernelsOnly=*/true);
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
  // With no result (a function returning its value through a pointer, or
  // nothing), `scf.if` comes with an empty `scf.yield` ending each branch
  // already, and its body builders insert before it: no yield of our own.
  {
    OpBuilder tb = ifOp.getThenBodyBuilder();
    auto direct = tb.create<func::CallOp>(loc, inner, TypeRange(results), args);
    if (!results.empty())
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
    if (!results.empty())
      eb.create<scf::YieldOp>(loc, loaded);
  }
  b.setInsertionPointToEnd(entry);
  b.create<func::ReturnOp>(loc, ifOp.getResults());
}

} // namespace

static bool cleaveLowerSpawns(MlirOperation op, bool tasks) {
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

// ------------------------------------------------------------------ passes
//
// Each rewrite above as a registered MLIR pass on the module, so a stage of
// cleave's pipeline is one textual pipeline (`cleaveRunPipeline`). What a
// rewrite counts is a pass statistic, printed when the pipeline runs with
// statistics on.

namespace {

// A pass running `rewrite` on the module, counting what it returns.
#define CLEAVE_COUNTING_PASS(Class, argument, description, rewrite)            \
  struct Class : PassWrapper<Class, OperationPass<ModuleOp>> {                 \
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(Class)                        \
    Class() = default;                                                         \
    Class(const Class &other) : PassWrapper(other) {}                          \
    StringRef getArgument() const final { return argument; }                   \
    StringRef getName() const final { return argument; }                       \
    StringRef getDescription() const final { return description; }           \
    Statistic rewritten{this, "rewritten", description};                       \
    void runOnOperation() final {                                              \
      rewritten += rewrite(wrap(getOperation().getOperation()));               \
    }                                                                          \
  };

// A pass running `rewrite` on the module, which reports nothing.
#define CLEAVE_PLAIN_PASS(Class, argument, description, rewrite)               \
  struct Class : PassWrapper<Class, OperationPass<ModuleOp>> {                 \
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(Class)                        \
    StringRef getArgument() const final { return argument; }                   \
    StringRef getName() const final { return argument; }                       \
    StringRef getDescription() const final { return description; }           \
    void runOnOperation() final {                                              \
      rewrite(wrap(getOperation().getOperation()));                            \
    }                                                                          \
  };

CLEAVE_COUNTING_PASS(LowerBlasMatmulsPass, "cleave-lower-blas-matmuls",
                     "BLAS-marked products lowered to sgemm calls",
                     cleaveLowerBlasMatmuls)
CLEAVE_COUNTING_PASS(ReuseDyingInputsPass, "cleave-reuse-dying-inputs",
                     "elementwise results written into a dying operand",
                     cleaveReuseDyingInputs)
CLEAVE_COUNTING_PASS(ElideBlockCopiesPass, "cleave-elide-block-copies",
                     "blocks written in place", cleaveElideBlockCopies)
CLEAVE_COUNTING_PASS(ForwardCopiesToDestinationsPass,
                     "cleave-forward-copies-to-destinations",
                     "results written in their destination",
                     cleaveForwardCopiesToDestinations)
struct TileParallelLoopsForGpuPass : PassWrapper<TileParallelLoopsForGpuPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TileParallelLoopsForGpuPass)
  TileParallelLoopsForGpuPass() = default;
  TileParallelLoopsForGpuPass(const TileParallelLoopsForGpuPass &other) : PassWrapper(other) {}
  StringRef getArgument() const final { return "cleave-tile-parallel-loops-for-gpu"; }
  StringRef getDescription() const final { return "parallel loops split into GPU workgroups along their widest dimension"; }
  Option<int64_t> threads{*this, "threads", llvm::cl::desc("invocations per workgroup"), llvm::cl::init(64)};
  Statistic tiled{this, "tiled", "parallel loops split into workgroups"};
  void runOnOperation() final { tiled += cleaveTileParallelLoopsForGpu(wrap(getOperation().getOperation()), threads); }
};
CLEAVE_COUNTING_PASS(SetSpirvEntryAbiPass, "cleave-set-spirv-entry-abi",
                     "GPU kernels given their SPIR-V workgroup size",
                     cleaveSetSpirvEntryAbi)
CLEAVE_COUNTING_PASS(ParallelIteratorsFirstPass, "cleave-parallel-iterators-first",
                     "linalg.generic ops interchanged to put their parallel iterators first",
                     cleaveParallelIteratorsFirst)
CLEAVE_COUNTING_PASS(MarkFullOverwritesPass, "cleave-mark-full-overwrites",
                     "linalg ops overwriting a whole heap payload, reported to the Vulkan runtime",
                     cleaveMarkFullOverwrites)
CLEAVE_COUNTING_PASS(GuardZeroTripLoopsPass, "cleave-guard-zero-trip-loops",
                     "kernel loops yielding values run only when they iterate",
                     cleaveGuardZeroTripLoops)
CLEAVE_COUNTING_PASS(SinkViewsIntoLaunchesPass, "cleave-sink-views-into-launches",
                     "GPU launches given the views they use",
                     cleaveSinkViewsIntoLaunches)
CLEAVE_COUNTING_PASS(BlasToLinalgPass, "cleave-blas-to-linalg",
                     "sgemm calls made linalg ops, for the GPU",
                     cleaveBlasToLinalg)
CLEAVE_COUNTING_PASS(ParallelizeLoopsPass, "cleave-parallelize-loops",
                     "loops whose iterations are independent made scf.parallel",
                     cleaveParallelizeLoops)
CLEAVE_COUNTING_PASS(GpuCopiesPass, "cleave-gpu-copies",
                     "copies between heap payloads made linalg copies, for the GPU",
                     cleaveGpuCopies)
CLEAVE_COUNTING_PASS(SyncHostAccessesPass, "cleave-sync-host-accesses",
                     "host accesses to memory kernels may share, reported to the Vulkan runtime",
                     cleaveSyncHostAccesses)
CLEAVE_COUNTING_PASS(LowerGpuLaunchesPass, "cleave-lower-gpu-launches",
                     "GPU kernel launches lowered to Vulkan runtime calls",
                     cleaveLowerGpuLaunches)
CLEAVE_COUNTING_PASS(DropZeroTripWhilesPass, "cleave-drop-zero-trip-whiles",
                     "loops whose condition is false on entry, replaced by one run of it",
                     cleaveDropZeroTripWhiles)
CLEAVE_COUNTING_PASS(ForwardReadOnlyArgumentsPass,
                     "cleave-forward-read-only-arguments",
                     "layout-only copies passed to read-only parameters, removed",
                     cleaveForwardReadOnlyArguments)
CLEAVE_COUNTING_PASS(FoldPassthroughIterArgsPass,
                     "cleave-fold-passthrough-iter-args",
                     "loop-carried values yielded back unchanged, removed",
                     cleaveFoldPassthroughIterArgs)
CLEAVE_COUNTING_PASS(DeallocAtLastUsePass, "cleave-dealloc-at-last-use",
                     "deallocations moved to the last use",
                     cleaveDeallocAtLastUse)
CLEAVE_COUNTING_PASS(LowerNonAffineLinalgPass, "cleave-lower-non-affine-linalg",
                     "linalg ops with non-affine bounds lowered to scf loops",
                     cleaveLowerNonAffineLinalg)
CLEAVE_COUNTING_PASS(BindTeamsPass, "cleave-bind-teams",
                     "parallel regions placing their threads", cleaveBindTeams)
CLEAVE_PLAIN_PASS(LowerAdoptionsPass, "cleave-lower-adoptions",
                  "adoption markers lowered", cleaveLowerAdoptions)
CLEAVE_PLAIN_PASS(HoistArgSlotsPass, "cleave-hoist-arg-slots",
                  "argument slots hoisted to the entry block, their lifetimes "
                  "bounded",
                  cleaveHoistArgSlots)
CLEAVE_PLAIN_PASS(ApplyNoInlinePass, "cleave-apply-no-inline",
                  "functions kept out of line stay out of line in LLVM",
                  cleaveApplyNoInline)

#undef CLEAVE_COUNTING_PASS
#undef CLEAVE_PLAIN_PASS

struct ApproximateMathPass
    : PassWrapper<ApproximateMathPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ApproximateMathPass)
  StringRef getArgument() const final { return "cleave-approximate-math"; }
  StringRef getName() const final { return "cleave-approximate-math"; }
  StringRef getDescription() const final {
    return "transcendentals rewritten as vectorizable polynomials";
  }
  void runOnOperation() final {
    if (!cleaveApproximateMath(wrap(getOperation().getOperation())))
      signalPassFailure();
  }
};

struct AccurateKernelMathPass : PassWrapper<AccurateKernelMathPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AccurateKernelMathPass)
  StringRef getArgument() const final { return "cleave-accurate-kernel-math"; }
  StringRef getName() const final { return "cleave-accurate-kernel-math"; }
  StringRef getDescription() const final { return "sin and cos in GPU kernels made accurate polynomials"; }
  void runOnOperation() final {
    if (!cleaveAccurateKernelMath(wrap(getOperation().getOperation())))
      signalPassFailure();
  }
};

struct LimitInliningPass
    : PassWrapper<LimitInliningPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LimitInliningPass)
  LimitInliningPass() = default;
  LimitInliningPass(const LimitInliningPass &other) : PassWrapper(other) {}
  StringRef getArgument() const final { return "cleave-limit-inlining"; }
  StringRef getName() const final { return "cleave-limit-inlining"; }
  StringRef getDescription() const final {
    return "calls kept out of line by the inline threshold";
  }
  Option<int64_t> threshold{*this, "threshold",
                            llvm::cl::desc("the inline threshold"),
                            llvm::cl::init(0)};
  Statistic kept{this, "kept", "calls kept out of line"};
  void runOnOperation() final {
    kept += cleaveLimitInlining(wrap(getOperation().getOperation()), threshold);
  }
};

struct BlasTileAndFusePass
    : PassWrapper<BlasTileAndFusePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BlasTileAndFusePass)
  BlasTileAndFusePass() = default;
  BlasTileAndFusePass(const BlasTileAndFusePass &other) : PassWrapper(other) {}
  StringRef getArgument() const final { return "cleave-blas-tile-and-fuse"; }
  StringRef getName() const final { return "cleave-blas-tile-and-fuse"; }
  StringRef getDescription() const final {
    return "BLAS products fused with their consumer, tiled by rows";
  }
  Option<int64_t> rows{*this, "rows", llvm::cl::desc("rows per tile"),
                       llvm::cl::init(128)};
  Statistic fused{this, "fused", "products fused with their consumer"};
  void runOnOperation() final {
    fused += cleaveBlasTileAndFuse(wrap(getOperation().getOperation()), rows);
  }
};

struct SplitRowRemaindersPass
    : PassWrapper<SplitRowRemaindersPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SplitRowRemaindersPass)
  SplitRowRemaindersPass() = default;
  SplitRowRemaindersPass(const SplitRowRemaindersPass &other) : PassWrapper(other) {}
  StringRef getArgument() const final { return "cleave-split-row-remainders"; }
  StringRef getName() const final { return "cleave-split-row-remainders"; }
  StringRef getDescription() const final {
    return "products split into whole row tiles and a remainder of static size";
  }
  Option<int64_t> rows{*this, "rows", llvm::cl::desc("rows per tile"),
                       llvm::cl::init(8)};
  Statistic split{this, "split", "operations split"};
  void runOnOperation() final {
    split += cleaveSplitRowRemainders(wrap(getOperation().getOperation()), rows);
  }
};

struct CopyAggregatesInMemoryPass
    : PassWrapper<CopyAggregatesInMemoryPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CopyAggregatesInMemoryPass)
  CopyAggregatesInMemoryPass() = default;
  CopyAggregatesInMemoryPass(const CopyAggregatesInMemoryPass &other)
      : PassWrapper(other) {}
  StringRef getArgument() const final {
    return "cleave-copy-aggregates-in-memory";
  }
  StringRef getName() const final { return "cleave-copy-aggregates-in-memory"; }
  StringRef getDescription() const final {
    return "large aggregates copied from memory to memory become memcpys";
  }
  Option<int64_t> minBytes{*this, "min-bytes",
                           llvm::cl::desc("the smallest aggregate copied so"),
                           llvm::cl::init(0)};
  Statistic copies{this, "copies", "aggregate copies made memcpy"};
  void runOnOperation() final {
    copies += cleaveCopyAggregatesInMemory(
        wrap(getOperation().getOperation()), minBytes);
  }
};

struct LowerSpawnsPass : PassWrapper<LowerSpawnsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerSpawnsPass)
  LowerSpawnsPass() = default;
  LowerSpawnsPass(const LowerSpawnsPass &other) : PassWrapper(other) {}
  StringRef getArgument() const final { return "cleave-lower-spawns"; }
  StringRef getName() const final { return "cleave-lower-spawns"; }
  StringRef getDescription() const final {
    return "spawn markers lowered to OpenMP tasks, or removed";
  }
  Option<bool> tasks{*this, "tasks",
                     llvm::cl::desc("OpenMP tasks; false runs each spawned "
                                    "call in place"),
                     llvm::cl::init(true)};
  void runOnOperation() final {
    if (!cleaveLowerSpawns(wrap(getOperation().getOperation()), tasks))
      signalPassFailure();
  }
};

} // namespace


// ------------------------------------------------------------------ rewrites
// of the bufferized and lowered module: copies, allocations, loop stack
// scopes, locations

namespace {

// Every operation's block and position in it, every value's uses (the using
// operation and the operand's index) and every `memref.copy`, in a pre-order
// walk of the module: what the copy rewrites below match against.
struct OpIndex {
  llvm::DenseMap<Operation *, std::pair<Block *, unsigned>> position;
  llvm::DenseMap<Value, SmallVector<std::pair<Operation *, unsigned>>> uses;
  SmallVector<Operation *> copies;
};

static void indexBlock(Block &block, OpIndex &index) {
  unsigned pos = 0;
  for (Operation &op : block) {
    index.position[&op] = {&block, pos};
    for (OpOperand &operand : op.getOpOperands())
      index.uses[operand.get()].push_back({&op, operand.getOperandNumber()});
    if (isa<memref::CopyOp>(op))
      index.copies.push_back(&op);
    for (Region &region : op.getRegions())
      for (Block &b : region)
        indexBlock(b, index);
    ++pos;
  }
}

static OpIndex indexModule(ModuleOp module) {
  OpIndex index;
  indexBlock(*module.getBody(), index);
  return index;
}

// The position in `block` of `user`'s ancestor there (itself included).
static std::optional<unsigned> positionIn(const OpIndex &index, Block *block,
                                          Operation *user) {
  Operation *ancestor = block->findAncestorOpInBlock(*user);
  if (!ancestor)
    return std::nullopt;
  auto it = index.position.find(ancestor);
  if (it == index.position.end())
    return std::nullopt;
  return it->second.second;
}

static ArrayRef<std::pair<Operation *, unsigned>> usesOf(const OpIndex &index,
                                                         Value v) {
  auto it = index.uses.find(v);
  if (it == index.uses.end())
    return {};
  return it->second;
}

// `memref.copy %a, %b`, `%a` and `%b` both operand-less `memref.alloc`s of the
// same type in the copy's block, `%a` with no use after the copy (nested uses
// counted at their enclosing operation in that block), `%b` with none before
// it: `%a`, `%b` and `%b`'s allocation. `Tensor(data: buf)` copies `buf`
// before marking the tensor `restrict` (`mlir_lower.rs::lower_tagged_struct_
// construct`), since `buf` could in general still be written afterwards;
// when it isn't, the copy is pure cost.
struct DeadSourceCopy {
  Value a, b;
  Operation *bAlloc;
};
static std::optional<DeadSourceCopy> matchDeadSourceCopy(const OpIndex &index,
                                                         Operation *copy) {
  Value a = copy->getOperand(0), b = copy->getOperand(1);
  if (a == b || a.getType() != b.getType())
    return std::nullopt;
  auto [block, copyPos] = index.position.lookup(copy);
  auto allocOf = [&](Value v) -> Operation * {
    Operation *op = v.getDefiningOp();
    if (!op || !isa<memref::AllocOp>(op) || op->getNumOperands() != 0)
      return nullptr;
    auto it = index.position.find(op);
    if (it == index.position.end() || it->second.first != block)
      return nullptr;
    return op;
  };
  if (!allocOf(a))
    return std::nullopt;
  Operation *bAlloc = allocOf(b);
  if (!bAlloc || !index.uses.count(a) || !index.uses.count(b))
    return std::nullopt;
  for (auto [user, operand] : usesOf(index, a)) {
    if (user == copy)
      continue;
    auto pos = positionIn(index, block, user);
    if (!pos || *pos >= copyPos)
      return std::nullopt;
  }
  for (auto [user, operand] : usesOf(index, b)) {
    if (user == copy)
      continue;
    auto pos = positionIn(index, block, user);
    if (!pos || *pos <= copyPos)
      return std::nullopt;
  }
  return DeadSourceCopy{a, b, bAlloc};
}

// What `matchHoistableDestination` may move earlier: no side effect but
// allocating, no region.
static bool isMovable(Operation *op) {
  if (op->getNumRegions() != 0)
    return false;
  StringRef name = op->getName().getStringRef();
  if (name == "llvm.getelementptr" || name == "llvm.ptrtoint" ||
      name == "llvm.insertvalue" || name == "llvm.extractvalue" ||
      name == "llvm.mlir.poison" || name == "llvm.mlir.undef" ||
      name == "llvm.mlir.zero" || name == "llvm.mlir.constant" ||
      name == "arith.constant" || name == "builtin.unrealized_conversion_cast")
    return true;
  if (auto call = dyn_cast<func::CallOp>(op))
    return call.getCallee() == "cleave_alloc_rc";
  return false;
}

// The other direction: `memref.copy %a, %b` where `%a` is a fresh
// `memref.alloc` filled earlier and dead after the copy, and `%b` a
// destination defined later (a tuple element's or struct field's storage,
// `mlir_lower.rs::build_tensor_descriptor_value`). When `%b`'s definition is
// a chain of movable operations depending only on values available before
// `%a`'s allocation, the chain moves up next to it and `%a`'s writers write
// `%b` directly. `%a`'s allocation, `%a`, `%b` and the chain, in block order.
struct HoistableDestination {
  Operation *aAlloc;
  Value a, b;
  SmallVector<Operation *> slice;
};
static std::optional<HoistableDestination>
matchHoistableDestination(const OpIndex &index, Operation *copy) {
  Value a = copy->getOperand(0), b = copy->getOperand(1);
  if (a == b || a.getType() != b.getType())
    return std::nullopt;
  auto [block, copyPos] = index.position.lookup(copy);
  Operation *aAlloc = a.getDefiningOp();
  if (!aAlloc || !isa<memref::AllocOp>(aAlloc) || aAlloc->getNumOperands() != 0)
    return std::nullopt;
  auto aIt = index.position.find(aAlloc);
  if (aIt == index.position.end() || aIt->second.first != block)
    return std::nullopt;
  unsigned aPos = aIt->second.second;
  if (!index.uses.count(a))
    return std::nullopt;
  for (auto [user, operand] : usesOf(index, a)) {
    if (user == copy)
      continue;
    auto pos = positionIn(index, block, user);
    if (!pos || *pos >= copyPos)
      return std::nullopt;
  }
  Operation *bDef = b.getDefiningOp();
  if (!bDef)
    return std::nullopt;
  auto bIt = index.position.find(bDef);
  if (bIt == index.position.end() || bIt->second.first != block)
    return std::nullopt;
  unsigned bPos = bIt->second.second;
  if (bPos <= aPos || bPos >= copyPos)
    return std::nullopt;
  for (auto [user, operand] : usesOf(index, b)) {
    if (user == copy)
      continue;
    auto pos = positionIn(index, block, user);
    if (!pos || *pos <= copyPos)
      return std::nullopt;
  }
  // The backward slice of `%b` after `%a`'s allocation, every op movable;
  // operands defined before `%a`'s allocation, or block arguments, are
  // available at the new position already.
  SmallVector<std::pair<unsigned, Operation *>> slice;
  llvm::DenseSet<Operation *> seen;
  SmallVector<Operation *> work{bDef};
  while (!work.empty()) {
    Operation *op = work.pop_back_val();
    if (!seen.insert(op).second)
      continue;
    auto it = index.position.find(op);
    if (it == index.position.end())
      return std::nullopt;
    if (it->second.first != block || it->second.second <= aPos)
      continue;
    if (!isMovable(op))
      return std::nullopt;
    slice.push_back({it->second.second, op});
    for (Value operand : op->getOperands())
      if (Operation *def = operand.getDefiningOp())
        work.push_back(def);
  }
  // Nothing outside the slice may use its results before the copy.
  llvm::DenseSet<Operation *> inSlice;
  for (auto [pos, op] : slice)
    inSlice.insert(op);
  for (auto [pos, op] : slice)
    for (Value result : op->getResults())
      for (auto [user, operand] : usesOf(index, result)) {
        if (inSlice.contains(user) || user == copy)
          continue;
        auto userPos = positionIn(index, block, user);
        if (!userPos || *userPos <= copyPos)
          return std::nullopt;
      }
  llvm::stable_sort(slice, [](auto &x, auto &y) { return x.first < y.first; });
  HoistableDestination match{aAlloc, a, b, {}};
  for (auto [pos, op] : slice)
    match.slice.push_back(op);
  return match;
}

// Drops the copies `matchDeadSourceCopy` and `matchHoistableDestination` find,
// until none is left. Each round rewrites a buffer at most once: a later
// rewrite could involve a value an earlier one replaces.
static int64_t forwardDeadSourceCopies(ModuleOp module) {
  int64_t total = 0;
  while (true) {
    OpIndex index = indexModule(module);
    llvm::DenseSet<Value> done;
    SmallVector<std::pair<Operation *, DeadSourceCopy>> rewrites;
    SmallVector<std::pair<Operation *, HoistableDestination>> hoists;
    for (Operation *copy : index.copies) {
      if (auto m = matchDeadSourceCopy(index, copy)) {
        if (done.insert(m->a).second && done.insert(m->b).second)
          rewrites.push_back({copy, *m});
      } else if (auto h = matchHoistableDestination(index, copy)) {
        if (done.insert(h->a).second && done.insert(h->b).second)
          hoists.push_back({copy, *h});
      }
    }
    if (rewrites.empty() && hoists.empty())
      return total;
    for (auto &[copy, m] : rewrites) {
      for (auto [user, operand] : usesOf(index, m.b))
        if (user != copy)
          user->setOperand(operand, m.a);
      copy->erase();
      m.bAlloc->erase();
      ++total;
    }
    for (auto &[copy, h] : hoists) {
      Operation *anchor = h.aAlloc->getNextNode();
      for (Operation *op : h.slice)
        op->moveBefore(anchor);
      for (auto [user, operand] : usesOf(index, h.a))
        if (user != copy)
          user->setOperand(operand, h.b);
      copy->erase();
      h.aAlloc->erase();
      ++total;
    }
  }
}

// After `buffer-results-to-out-params`, a call whose result goes into a struct
// field or a tuple element allocates `%out`, passes it to the call, then
// copies it into the field: `call @f(%a, %out) ... memref.copy %out, %field`.
// Rewritten to `call @f(%a, %field)`, the copy and the allocation gone.
// Conservative, every condition local: `%out` is a `memref.alloc` used exactly
// twice, by one `func.call` and as the copy's source, all three in one block
// in that order; `%field` has the same type, is defined earlier in that block
// (so it dominates the call), and has no use before the copy, the call's own
// operands included. Runs `forwardDeadSourceCopies` first.
static int64_t forwardOutParamCopies(ModuleOp module) {
  int64_t total = forwardDeadSourceCopies(module);
  OpIndex index = indexModule(module);
  struct Rewrite {
    Operation *call;
    unsigned operand;
    Value dst;
    Operation *copy, *alloc;
  };
  SmallVector<Rewrite> rewrites;
  for (Operation *copy : index.copies) {
    Value src = copy->getOperand(0), dst = copy->getOperand(1);
    if (src == dst || src.getType() != dst.getType())
      continue;
    auto [block, copyPos] = index.position.lookup(copy);
    Operation *alloc = src.getDefiningOp();
    if (!alloc || !isa<memref::AllocOp>(alloc))
      continue;
    auto allocIt = index.position.find(alloc);
    if (allocIt == index.position.end() || !index.uses.count(src))
      continue;
    auto srcUses = usesOf(index, src);
    if (allocIt->second.first != block || srcUses.size() != 2)
      continue;
    auto other = llvm::find_if(srcUses, [&](auto &u) { return u.first != copy; });
    if (other == srcUses.end())
      continue;
    auto [call, operand] = *other;
    if (!isa<func::CallOp>(call))
      continue;
    auto callIt = index.position.find(call);
    if (callIt == index.position.end() || callIt->second.first != block)
      continue;
    unsigned allocPos = allocIt->second.second, callPos = callIt->second.second;
    if (!(allocPos < callPos && callPos < copyPos))
      continue;
    Operation *def = dst.getDefiningOp();
    if (!def)
      continue;
    auto defIt = index.position.find(def);
    if (defIt == index.position.end() || defIt->second.first != block ||
        defIt->second.second >= callPos)
      continue;
    bool usedBefore = false;
    for (auto [user, i] : usesOf(index, dst)) {
      if (user == copy)
        continue;
      auto pos = positionIn(index, block, user);
      if (!pos || *pos <= copyPos) {
        usedBefore = true;
        break;
      }
    }
    if (!usedBefore)
      rewrites.push_back({call, operand, dst, copy, alloc});
  }
  for (Rewrite &r : rewrites) {
    r.call->setOperand(r.operand, r.dst);
    r.copy->erase();
    r.alloc->erase();
    ++total;
  }
  return total;
}

// A `memref.copy %x, %x`, left by One-Shot Bufferize's write-back of a tiled
// `scf.forall` (`tensor.parallel_insert_slice`): a no-op, erased. Neither
// `canonicalize` nor `cse` folds it. Runs after `cse`, which merges the two
// identical subviews such a copy is made of.
static int64_t eliminateSelfCopies(ModuleOp module) {
  SmallVector<memref::CopyOp> dead;
  module.walk([&](memref::CopyOp copy) {
    if (copy.getSource() == copy.getTarget())
      dead.push_back(copy);
  });
  for (memref::CopyOp copy : dead)
    copy->erase();
  return dead.size();
}

// Every `linalg.copy` between memrefs of a dynamic size (or layout) made the
// `memref.copy` it is, before `convert-linalg-to-affine-loops`: the
// write-back of a partial tile (a matmul whose column count isn't a multiple
// of the schedule's 16), whose size is an `affine.min` of an `scf.for`
// induction variable, which the affine pass rejects as a dimension.
static int64_t lowerDynamicCopies(ModuleOp module) {
  SmallVector<Operation *> copies;
  module.walk([&](Operation *op) {
    if (op->getName().getStringRef() != "linalg.copy" ||
        op->getNumOperands() != 2 || op->getNumResults() != 0)
      return;
    for (Value v : op->getOperands()) {
      if (!isa<MemRefType>(v.getType()))
        continue;
      std::string text;
      llvm::raw_string_ostream os(text);
      v.getType().print(os);
      if (StringRef(text).contains('?')) {
        copies.push_back(op);
        return;
      }
    }
  });
  for (Operation *op : copies) {
    OpBuilder builder(op);
    memref::CopyOp::create(builder, op->getLoc(), op->getOperand(0),
                           op->getOperand(1));
    op->erase();
  }
  return copies.size();
}

// Every `llvm.call @old` renamed `@new`; `@old`'s declaration renamed too
// when nothing declares `@new` yet (same signature by construction).
static int64_t retargetCalls(ModuleOp module, StringRef oldName,
                             StringRef newName) {
  auto findDecl = [&](StringRef name) -> LLVM::LLVMFuncOp {
    for (Operation &op : *module.getBody())
      if (auto f = dyn_cast<LLVM::LLVMFuncOp>(op))
        if (f.getSymName() == name)
          return f;
    return nullptr;
  };
  bool newDeclared = static_cast<bool>(findDecl(newName));
  int64_t renamed = 0;
  auto callee = FlatSymbolRefAttr::get(module.getContext(), newName);
  module.walk([&](LLVM::CallOp call) {
    if (call.getCallee() == oldName) {
      call.setCalleeAttr(callee);
      ++renamed;
    }
  });
  if (renamed && !newDeclared)
    if (LLVM::LLVMFuncOp decl = findDecl(oldName))
      decl->setAttr("sym_name", StringAttr::get(module.getContext(), newName));
  return renamed;
}

// Every allocation MLIR's bufferization left as `malloc`/`free` (by now, every
// one is a tensor payload's: cleave's own code allocates through
// `cleave_alloc_rc`) goes through cleave's allocator instead:
// `@cleave_alloc_rc`/`@cleave_release_void`, same signatures. Runs once
// everything is in the LLVM dialect, where the swap is a rename.
static int64_t unifyTensorAllocations(ModuleOp module) {
  return retargetCalls(module, "malloc", "cleave_alloc_rc") +
         retargetCalls(module, "free", "cleave_release_void");
}

// `fastmath<contract>` on every `arith.mulf`/`arith.addf`: the permission
// LLVM needs to fuse a multiply and an add into one FMA.
static int64_t markMulfAddfContract(ModuleOp module) {
  auto contract = arith::FastMathFlagsAttr::get(module.getContext(),
                                                arith::FastMathFlags::contract);
  int64_t marked = 0;
  module.walk([&](Operation *op) {
    if (isa<arith::MulFOp, arith::AddFOp>(op)) {
      op->setAttr("fastmath", contract);
      ++marked;
    }
  });
  return marked;
}

// Every loop body (`scf.while`'s after region, `scf.for`'s and
// `scf.parallel`'s body) between `llvm.intr.stacksave` and
// `llvm.intr.stackrestore`: what a body allocates on the stack is freed at
// each iteration, as the arena region `mlir_lower.rs::lower_loop` opens per
// iteration frees what it allocates on the heap. Without it a training loop
// built without OpenMP (whose outlined bodies return, freeing their frames)
// overflowed the stack within a few iterations.
static int64_t insertStackScopesInLoops(ModuleOp module) {
  SmallVector<std::pair<Operation *, unsigned>> loops;
  module.walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (isa<scf::WhileOp>(op))
      loops.push_back({op, 1});
    else if (isa<scf::ForOp, scf::ParallelOp>(op))
      loops.push_back({op, 0});
  });
  MLIRContext *ctx = module.getContext();
  auto ptrTy = LLVM::LLVMPointerType::get(ctx);
  auto loc = UnknownLoc::get(ctx);
  int64_t scoped = 0;
  for (auto [loop, index] : loops) {
    Region &region = loop->getRegion(index);
    if (region.empty() || region.front().empty())
      continue;
    Block &body = region.front();
    OpBuilder builder(&body.front());
    Value saved = LLVM::StackSaveOp::create(builder, loc, ptrTy);
    Operation *last = &body.back();
    if (last->hasTrait<OpTrait::IsTerminator>()) {
      builder.setInsertionPoint(last);
      LLVM::StackRestoreOp::create(builder, loc, saved);
    }
    ++scoped;
  }
  return scoped;
}

// The `_mlir_ciface_*` wrappers `llvm.emit_c_interface` makes reuse their
// function's location, `DISubprogram` included, which LLVM's verifier
// rejects ("DISubprogram attached to more than one function"): their whole
// body gets an unknown location.
static void stripCifaceDebugInfo(ModuleOp module) {
  auto unknown = UnknownLoc::get(module.getContext());
  module.walk([&](LLVM::LLVMFuncOp f) {
    if (f.getSymName().contains("_mlir_ciface_"))
      f->walk([&](Operation *op) { op->setLoc(unknown); });
  });
}

// Whether `loc` resolves to a source line anywhere inside it (`--inline`
// wraps a cloned op's location in a `CallSiteLoc`, the line one level down).
static bool hasRealLine(Location loc) {
  if (isa<FileLineColRange>(loc))
    return true;
  if (auto callSite = dyn_cast<CallSiteLoc>(loc))
    return hasRealLine(callSite.getCallee()) || hasRealLine(callSite.getCaller());
  if (auto fused = dyn_cast<FusedLoc>(loc))
    return llvm::any_of(fused.getLocations(),
                        [](Location l) { return hasRealLine(l); });
  return false;
}

// In every `llvm.func` with a body, each operation without a source line (one
// an MLIR lowering synthesized: a tile's seed, a vector epilogue) takes the
// nearest real location before it in program order, the function's own to
// start with: a profiler or debugger then places it next to the code it was
// generated for, not on the function's declaration.
static void backfillUnknownLocations(ModuleOp module) {
  module.walk([&](LLVM::LLVMFuncOp f) {
    if (f.getBody().empty())
      return;
    Location current = f.getLoc();
    f.getBody().walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (hasRealLine(op->getLoc()))
        current = op->getLoc();
      else
        op->setLoc(current);
    });
  });
}

// Runs `pipeline` (nested in a module) on `module` if `condition` holds.
static LogicalResult runIf(ModuleOp module, bool condition,
                           StringRef pipeline,
                           function_ref<LogicalResult(OpPassManager &, Operation *)>
                               runPipeline) {
  if (!condition)
    return success();
  OpPassManager pm("builtin.module");
  if (failed(parsePassPipeline(pipeline, pm)))
    return failure();
  return runPipeline(pm, module);
}

// Whether `op` is a `vector.transfer_read`/`transfer_write` whose permutation
// map isn't a minor identity (`(d0, d1) -> (d1)` is one; `(d0, d1) -> (d0)`, a
// column, isn't): what `convert-vector-to-llvm` can't lower.
static bool isPermutedTransfer(Operation *op) {
  if (auto read = dyn_cast<vector::TransferReadOp>(op))
    return !read.getPermutationMap().isMinorIdentity();
  if (auto write = dyn_cast<vector::TransferWriteOp>(op))
    return !write.getPermutationMap().isMinorIdentity();
  return false;
}

// The operations a rewrite inserts, minus those it erases again.
struct TrackInserted : RewriterBase::Listener {
  llvm::SetVector<Operation *> ops;
  void notifyOperationInserted(Operation *op,
                               OpBuilder::InsertPoint) override {
    ops.insert(op);
  }
  void notifyOperationErased(Operation *op) override { ops.remove(op); }
};

// The permuted transfers of `module` lowered as `convert-vector-to-scf
// {target-rank=0}` lowers transfers (their permutation maps first, then
// scalar loops over a stack buffer), and only them and what lowering them
// creates: the pass itself would scalarize every transfer of the module, the
// vectorized loops around included, once a single permuted one exists (a
// training loop's matmuls went 3x slower when inlining put a column read in
// their function). Returns how many there were.
static int64_t lowerPermutedTransfers(ModuleOp module) {
  SmallVector<Operation *> permuted;
  module.walk([&](Operation *op) {
    if (isPermutedTransfer(op))
      permuted.push_back(op);
  });
  if (permuted.empty())
    return 0;
  MLIRContext *ctx = module.getContext();
  TrackInserted track;
  llvm::SetVector<Operation *> alive(permuted.begin(), permuted.end());
  struct Forget : RewriterBase::Listener {
    llvm::SetVector<Operation *> &alive;
    TrackInserted &track;
    Forget(llvm::SetVector<Operation *> &alive, TrackInserted &track)
        : alive(alive), track(track) {}
    void notifyOperationInserted(Operation *op,
                                 OpBuilder::InsertPoint ip) override {
      track.notifyOperationInserted(op, ip);
    }
    void notifyOperationErased(Operation *op) override {
      alive.remove(op);
      track.notifyOperationErased(op);
    }
  } listener(alive, track);

  RewritePatternSet mapPatterns(ctx);
  vector::populateVectorTransferPermutationMapLoweringPatterns(mapPatterns);
  (void)applyOpPatternsGreedily(
      alive.getArrayRef(), std::move(mapPatterns),
      GreedyRewriteConfig()
          .setStrictness(GreedyRewriteStrictness::ExistingAndNewOps)
          .setListener(&listener));

  SmallVector<Operation *> transfers;
  for (Operation *op : alive)
    transfers.push_back(op);
  for (Operation *op : track.ops)
    if (isa<vector::TransferReadOp, vector::TransferWriteOp>(op))
      transfers.push_back(op);
  RewritePatternSet scfPatterns(ctx);
  VectorTransferToSCFOptions options;
  options.targetRank = 0;
  populateVectorToSCFConversionPatterns(scfPatterns, options);
  (void)applyOpPatternsGreedily(
      transfers, std::move(scfPatterns),
      GreedyRewriteConfig().setStrictness(
          GreedyRewriteStrictness::ExistingAndNewOps));
  return permuted.size();
}

#define CLEAVE_MODULE_PASS(Class, argument, description, body)                 \
  struct Class : PassWrapper<Class, OperationPass<ModuleOp>> {                 \
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(Class)                        \
    Class() = default;                                                         \
    Class(const Class &other) : PassWrapper(other) {}                          \
    StringRef getArgument() const final { return argument; }                   \
    StringRef getName() const final { return argument; }                       \
    StringRef getDescription() const final { return description; }           \
    Statistic rewritten{this, "rewritten", description};                       \
    void runOnOperation() final { body }                                       \
  };

CLEAVE_MODULE_PASS(ForwardDeadSourceCopiesPass,
                   "cleave-forward-dead-source-copies",
                   "copies of a buffer dead after them into a fresh one, dropped",
                   rewritten += forwardDeadSourceCopies(getOperation());)
CLEAVE_MODULE_PASS(ForwardOutParamCopiesPass, "cleave-forward-out-param-copies",
                   "calls writing their result directly where it was copied",
                   rewritten += forwardOutParamCopies(getOperation());)
CLEAVE_MODULE_PASS(EliminateSelfCopiesPass, "cleave-eliminate-self-copies",
                   "copies of a buffer to itself, erased",
                   rewritten += eliminateSelfCopies(getOperation());)
CLEAVE_MODULE_PASS(LowerDynamicCopiesPass, "cleave-lower-dynamic-copies",
                   "linalg.copy of a dynamic size made memref.copy",
                   rewritten += lowerDynamicCopies(getOperation());)
CLEAVE_MODULE_PASS(UnifyTensorAllocationsPass,
                   "cleave-unify-tensor-allocations",
                   "malloc/free calls made cleave allocator calls",
                   rewritten += unifyTensorAllocations(getOperation());)
CLEAVE_MODULE_PASS(MarkContractPass, "cleave-mark-contract",
                   "mulf/addf allowed to contract into an FMA",
                   rewritten += markMulfAddfContract(getOperation());)
CLEAVE_MODULE_PASS(InsertStackScopesPass, "cleave-insert-stack-scopes",
                   "loop bodies freeing their stack at each iteration",
                   rewritten += insertStackScopesInLoops(getOperation());)
CLEAVE_MODULE_PASS(StripCifaceDebugInfoPass, "cleave-strip-ciface-debug-info",
                   "C-interface wrappers given no debug location",
                   stripCifaceDebugInfo(getOperation());)
CLEAVE_MODULE_PASS(BackfillLocationsPass, "cleave-backfill-locations",
                   "synthesized operations given their nearest source line",
                   backfillUnknownLocations(getOperation());)
CLEAVE_MODULE_PASS(LowerPermutedTransfersPass,
                   "cleave-lower-permuted-transfers",
                   "permuted vector transfers lowered to scalar loops",
                   rewritten += lowerPermutedTransfers(getOperation());)

#undef CLEAVE_MODULE_PASS

// `convert-openmp-to-llvm`, when the module has OpenMP operations or `always`
// (`--openmp`'s parallel loops, or the tasks `spawn` lowers to).
struct ConvertOpenMPIfUsedPass
    : PassWrapper<ConvertOpenMPIfUsedPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertOpenMPIfUsedPass)
  ConvertOpenMPIfUsedPass() = default;
  ConvertOpenMPIfUsedPass(const ConvertOpenMPIfUsedPass &other)
      : PassWrapper(other) {}
  StringRef getArgument() const final { return "cleave-convert-openmp-if-used"; }
  StringRef getName() const final { return "cleave-convert-openmp-if-used"; }
  StringRef getDescription() const final {
    return "convert-openmp-to-llvm, when OpenMP operations are present";
  }
  Option<bool> always{*this, "always",
                      llvm::cl::desc("convert even with no OpenMP operation"),
                      llvm::cl::init(false)};
  void runOnOperation() final {
    bool used = always;
    getOperation().walk([&](Operation *op) {
      if (op->getName().getDialectNamespace() == "omp") {
        used = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (failed(runIf(getOperation(), used, "convert-openmp-to-llvm",
                     [&](OpPassManager &pm, Operation *op) {
                       return runPipeline(pm, op);
                     })))
      signalPassFailure();
  }
};

} // namespace

// Registers the passes above; called by `cleaveRegisterPasses`.
static void registerRewritePasses() {
  PassRegistration<ForwardDeadSourceCopiesPass>();
  PassRegistration<ForwardOutParamCopiesPass>();
  PassRegistration<EliminateSelfCopiesPass>();
  PassRegistration<LowerDynamicCopiesPass>();
  PassRegistration<UnifyTensorAllocationsPass>();
  PassRegistration<MarkContractPass>();
  PassRegistration<InsertStackScopesPass>();
  PassRegistration<StripCifaceDebugInfoPass>();
  PassRegistration<BackfillLocationsPass>();
  PassRegistration<LowerPermutedTransfersPass>();
  PassRegistration<ConvertOpenMPIfUsedPass>();
}

// Registers cleave's passes (above) and every MLIR pass, for pipelines to
// name them. Idempotent.
extern "C" void cleaveRegisterPasses() {
  static bool once = [] {
    registerAllPasses();
    PassRegistration<LimitInliningPass>();
    PassRegistration<BlasTileAndFusePass>();
    PassRegistration<SplitRowRemaindersPass>();
    PassRegistration<LowerNonAffineLinalgPass>();
    PassRegistration<LowerBlasMatmulsPass>();
    PassRegistration<ReuseDyingInputsPass>();
    PassRegistration<ElideBlockCopiesPass>();
    PassRegistration<ForwardCopiesToDestinationsPass>();
    PassRegistration<ForwardReadOnlyArgumentsPass>();
    PassRegistration<DropZeroTripWhilesPass>();
    PassRegistration<TileParallelLoopsForGpuPass>();
    PassRegistration<SetSpirvEntryAbiPass>();
    PassRegistration<LowerGpuLaunchesPass>();
    PassRegistration<SyncHostAccessesPass>();
    PassRegistration<GpuCopiesPass>();
    PassRegistration<ParallelizeLoopsPass>();
    PassRegistration<BlasToLinalgPass>();
    PassRegistration<MarkFullOverwritesPass>();
    PassRegistration<SinkViewsIntoLaunchesPass>();
    PassRegistration<GuardZeroTripLoopsPass>();
    PassRegistration<ParallelIteratorsFirstPass>();
    PassRegistration<FoldPassthroughIterArgsPass>();
    PassRegistration<LowerAdoptionsPass>();
    PassRegistration<DeallocAtLastUsePass>();
    PassRegistration<LowerSpawnsPass>();
    PassRegistration<ApproximateMathPass>();
    PassRegistration<AccurateKernelMathPass>();
    PassRegistration<BindTeamsPass>();
    PassRegistration<CopyAggregatesInMemoryPass>();
    PassRegistration<HoistArgSlotsPass>();
    PassRegistration<ApplyNoInlinePass>();
    registerRewritePasses();
    return true;
  }();
  (void)once;
}

// Runs the textual pipeline `pipeline` (`builtin.module(...)`) on `module`,
// with the pass statistics printed on stderr if `statistics`. `false` if the
// pipeline doesn't parse (reported through `onError`) or a pass fails (its
// diagnostics already reported by the context's handlers).
extern "C" bool cleaveRunPipeline(MlirModule module, MlirStringRef pipeline,
                                  bool statistics, CleaveErrorCallback onError,
                                  void *userData) {
  cleaveRegisterPasses();
  ModuleOp op = unwrap(module);
  std::string parseErrors;
  llvm::raw_string_ostream errorStream(parseErrors);
  FailureOr<OpPassManager> parsed = parsePassPipeline(unwrap(pipeline), errorStream);
  if (failed(parsed)) {
    report(onError, userData, parseErrors);
    return false;
  }
  PassManager pm(op->getContext(), parsed->getOpAnchorName());
  static_cast<OpPassManager &>(pm) = std::move(*parsed);
  if (statistics)
    pm.enableStatistics();
  return succeeded(pm.run(op));
}

// Loads the transform module `text` (named `name` in diagnostics) into the
// transform dialect's library for `context`, where `transform-interpreter`
// finds the sequences it names: what `transform-preload-library` does with a
// file, from memory. `false` on error, reported through the context's
// diagnostic handlers.
extern "C" bool cleaveLoadTransformLibrary(MlirContext context,
                                           MlirStringRef text,
                                           MlirStringRef name) {
  MLIRContext *ctx = unwrap(context);
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBufferCopy(unwrap(text), unwrap(name)),
      llvm::SMLoc());
  OwningOpRef<ModuleOp> library = parseSourceFile<ModuleOp>(sourceMgr, ctx);
  if (!library || failed(mlir::verify(*library)))
    return false;
  // Loaded already (a second compile in the same context): its sequences are
  // in the library; merging them again would define each one twice.
  auto *dialect = ctx->getOrLoadDialect<transform::TransformDialect>();
  if (ModuleOp existing = dialect->getLibraryModule()) {
    bool loaded = llvm::all_of(library->getBody()->getOps<SymbolOpInterface>(), [&](SymbolOpInterface symbol) {
      return SymbolTable::lookupSymbolIn(existing, symbol.getNameAttr()) != nullptr;
    });
    if (loaded)
      return true;
  }
  auto loc = FileLineColLoc::get(ctx, "<shared-library-module>", 0, 0);
  OwningOpRef<ModuleOp> merged = ModuleOp::create(loc, "__transform");
  merged.get()->setAttr("transform.with_named_sequence", UnitAttr::get(ctx));
  if (failed(transform::detail::mergeSymbolsInto(merged.get(),
                                                 std::move(library))))
    return false;
  return succeeded(ctx->getOrLoadDialect<transform::TransformDialect>()
                       ->loadIntoLibraryModule(std::move(merged)));
}

// The SPIR-V binary of the `spirv.module` in `module` named `name` (the only
// one when `name` is empty): what a Vulkan driver takes, as 32-bit words,
// handed to `onWords`. `false` on error, reported to `onError`.
using CleaveWordsCallback = void (*)(const uint32_t *, size_t, void *);
extern "C" bool cleaveSerializeSpirv(MlirModule module, MlirStringRef name,
                                     CleaveWordsCallback onWords,
                                     CleaveErrorCallback onError,
                                     void *userData) {
  StringRef wanted = unwrap(name);
  SmallVector<spirv::ModuleOp> found;
  unwrap(module)->walk([&](spirv::ModuleOp spirvModule) {
    if (wanted.empty() || spirvModule.getName() == wanted)
      found.push_back(spirvModule);
  });
  if (found.size() != 1) {
    report(onError, userData,
           (found.empty() ? "no spirv.module" : "several spirv.module") +
               (wanted.empty() ? std::string() : " named " + wanted.str()));
    return false;
  }
  SmallVector<uint32_t> words;
  if (failed(spirv::serialize(found.front(), words))) {
    report(onError, userData, "the spirv.module doesn't serialize");
    return false;
  }
  onWords(words.data(), words.size(), userData);
  return true;
}
