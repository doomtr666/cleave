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
#include "mlir/Dialect/Utils/StaticValueUtils.h"
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
// The host's CPU name (`llvm::sys::getHostCPUName`, what `native` means),
// copied into `buffer` (`size` bytes, NUL-terminated, truncated if needed).
// Returns its full length.
extern "C" size_t cleaveHostCpuName(char *buffer, size_t size) {
  std::string name = llvm::sys::getHostCPUName().str();
  if (size > 0) {
    size_t n = std::min(size - 1, name.size());
    std::memcpy(buffer, name.data(), n);
    buffer[n] = 0;
  }
  return name.size();
}

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
extern "C" int64_t cleaveLimitInlining(MlirOperation op, int64_t threshold) {
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

extern "C" int64_t cleaveCopyAggregatesInMemory(MlirOperation op, int64_t minBytes) {
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
extern "C" void cleaveApplyNoInline(MlirOperation op) {
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
extern "C" int64_t cleaveElideBlockCopies(MlirOperation op) {
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
extern "C" int64_t cleaveForwardCopiesToDestinations(MlirOperation op) {
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

// A product BLAS computes (`linalg.matmul` marked `cleave.blas`,
// `stdlib/linalg/matrix.cleave` above `BLAS_MIN_WORK`) whose only consumer
// is an elementwise op (a bias, an activation, a residual, a gradient's
// pointwise factor): the consumer is tiled by `rows` rows and the product,
// with what initializes it (its zero fill, a broadcast bias), fused into the
// same loop. Each tile of the product is then consumed while it is still in
// L2 instead of written to memory whole and read back: with every core busy,
// memory gives a core ~6 GB/s where its L2 gives ~220. Before
// `cleaveLowerBlasMatmuls`, on tensors. Returns how many products it fused.
extern "C" int64_t cleaveBlasTileAndFuse(MlirOperation op, int64_t rows) {
  // A consumer of several products (SwiGLU's `silu(gate) * up`) once: all
  // of them are fused into its loop.
  llvm::SetVector<linalg::GenericOp> consumers;
  unwrap(op)->walk([&](linalg::MatmulOp product) {
    // Only what `cleaveLowerBlasMatmuls` will make a `sgemm` (`f32`).
    if (!product->hasAttr("cleave.blas") || !product.hasPureTensorSemantics() ||
        !getElementTypeOrSelf(product->getResult(0).getType()).isF32())
      return;
    Value result = product->getResult(0);
    if (!result.hasOneUse())
      return;
    auto consumer = dyn_cast<linalg::GenericOp>(*result.getUsers().begin());
    if (!consumer || !consumer.hasPureTensorSemantics() || consumer.getNumLoops() != 2 ||
        consumer.getNumParallelLoops() != 2 || consumer.getNumDpsInits() != 1)
      return;
    // Pointwise over the product's rows: every operand read through a
    // projected permutation, the result written through the identity, the
    // product read through the identity.
    if (!llvm::all_of(consumer.getIndexingMapsArray(), [](AffineMap m) { return m.isProjectedPermutation(); }) ||
        !consumer.getMatchingIndexingMap(consumer.getDpsInitOperand(0)).isIdentity())
      return;
    for (OpOperand &input : consumer->getOpOperands())
      if (input.get() == result && !consumer.getMatchingIndexingMap(&input).isIdentity())
        return;
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
extern "C" int64_t cleaveLowerBlasMatmuls(MlirOperation op) {
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
extern "C" int64_t cleaveReuseDyingInputs(MlirOperation op) {
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
extern "C" int64_t cleaveFoldPassthroughIterArgs(MlirOperation op) {
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

extern "C" int64_t cleaveDeallocAtLastUse(MlirOperation op) {
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
extern "C" int64_t cleaveBindTeams(MlirOperation op) {
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
// is reference counted (`unify_alloc.rs`), so that release only drops the
// reference the retain added.
extern "C" void cleaveLowerAdoptions(MlirOperation op) {
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
