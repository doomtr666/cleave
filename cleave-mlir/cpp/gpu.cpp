// The Vulkan backend's passes (`doc/plan-vulkan.md`): loops made kernels,
// the transfers between host and device made explicit, kernels lowered to
// SPIR-V and to calls of the runtime (`cleave-vk`).

#include "cleave.h"

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
    scf::tileParallelLoop(loop, sizes, /*noMinMaxBounds=*/false);
    ++tiled;
  }
  if (tiled == 0)
    return 0;
  // The steps and bounds tiling computes (`step * size`) folded to
  // constants: `convert-parallel-loops-to-gpu` needs a workgroup size it can
  // read, and leaves a loop whose size it can't on the host. Folded here, not
  // by `canonicalize`, which would also merge the two nested loops into one,
  // all of it then mapped to blocks of one thread. `affine.min`'s constant
  // operands folded into its map too, so that the conversion can bound the
  // inner loop by the tile size. Once, after every loop is tiled: per loop,
  // over its parent, it was quadratic (146 s of nanoLM's compile once the
  // tiled products' kernels were in those parents).
  MLIRContext *ctx = unwrap(op)->getContext();
  RewritePatternSet patterns(ctx);
  affine::AffineMinOp::getCanonicalizationPatterns(patterns, ctx);
  arith::MulIOp::getCanonicalizationPatterns(patterns, ctx);
  arith::AddIOp::getCanonicalizationPatterns(patterns, ctx);
  SmallVector<Operation *> scalar;
  unwrap(op)->walk([&](Operation *inner) {
    if (isa<arith::MulIOp, arith::AddIOp, arith::SubIOp, affine::AffineMinOp, arith::ConstantOp>(inner) &&
        !inner->getParentOfType<gpu::LaunchOp>())
      scalar.push_back(inner);
  });
  (void)applyOpPatternsGreedily(scalar, FrozenRewritePatternSet(std::move(patterns)));
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
// What `isFreshUnaliased` remembers: its answers, and every function's
// call sites, indexed once (a walk of the whole module per question made
// `cleave-mark-full-overwrites` take seconds on nanoLM). `clear` after
// changing the IR.
struct FreshMemo {
  llvm::DenseMap<Value, bool> answers;
  std::unique_ptr<SymbolTableCollection> tables;
  std::unique_ptr<SymbolUserMap> users;
  void clear() {
    answers.clear();
    users.reset();
    tables.reset();
  }
};
static bool isFreshUnaliased(Value v, ModuleOp module, FreshMemo &memo);

static bool iterationsAreIndependent(scf::ForOp loop, ModuleOp module, FreshMemo &fresh) {
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
  FreshMemo fresh;
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

static bool slotsOnlyCompared(Value slots, ModuleOp module, int depth);

// Whether the integer `address` (a memref's address, as an index) is only
// compared, never followed: `arith.cmpi`s, through casts, or stored into a
// buffer of addresses whose loads are only compared in turn
// (`slotsOnlyCompared`). Bufferization's deallocation does that with the
// buffers a loop carries, every iteration, to decide which alias
// (`dealloc_helper`): reported as accesses, they marked nanoLM's attention
// gradients stale at each head, 14 GiB uploaded a step.
static bool addressOnlyCompared(Value address, ModuleOp module, int depth) {
  if (depth == 0)
    return false;
  SmallVector<Value> pending{address};
  while (!pending.empty()) {
    Value v = pending.pop_back_val();
    for (OpOperand &use : v.getUses()) {
      Operation *user = use.getOwner();
      if (isa<arith::CmpIOp>(user))
        continue;
      if (isa<arith::IndexCastOp>(user)) {
        pending.push_back(user->getResult(0));
        continue;
      }
      auto store = dyn_cast<memref::StoreOp>(user);
      if (store && use.getOperandNumber() == 0 && slotsOnlyCompared(store.getMemRef(), module, depth - 1))
        continue;
      return false;
    }
  }
  return true;
}

// Whether what's loaded from the memref `slots` is only compared
// (`addressOnlyCompared`): every use a store into it, a load whose value is
// only compared, its metadata, its deallocation, a view of it, the
// runtime's own address of it, or a call passing it to a function of the
// module whose parameter is used the same way.
static bool slotsOnlyCompared(Value slots, ModuleOp module, int depth) {
  if (depth == 0)
    return false;
  SmallVector<Value> pending{slots};
  while (!pending.empty()) {
    Value v = pending.pop_back_val();
    for (OpOperand &use : v.getUses()) {
      Operation *user = use.getOwner();
      if (isa<memref::DimOp, memref::DeallocOp>(user))
        continue;
      if (auto store = dyn_cast<memref::StoreOp>(user); store && use.getOperandNumber() == 1)
        continue;
      if (auto load = dyn_cast<memref::LoadOp>(user)) {
        if (!addressOnlyCompared(load.getResult(), module, depth - 1))
          return false;
        continue;
      }
      if (auto extract = dyn_cast<memref::ExtractAlignedPointerAsIndexOp>(user); extract &&
                                                                                   extract->hasAttr(kRuntimeAddress))
        continue;
      if (isa<ViewLikeOpInterface, memref::CastOp>(user)) {
        pending.push_back(user->getResult(0));
        continue;
      }
      if (auto metadata = dyn_cast<memref::ExtractStridedMetadataOp>(user)) {
        pending.push_back(metadata.getBaseBuffer());
        continue;
      }
      if (auto call = dyn_cast<func::CallOp>(user)) {
        auto callee = module.lookupSymbol<func::FuncOp>(call.getCallee());
        if (callee && !callee.isExternal() &&
            slotsOnlyCompared(callee.getArgument(use.getOperandNumber()), module, depth - 1))
          continue;
      }
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
        if (!extract->hasAttr(kRuntimeAddress) && !(onlyBuildsDescriptors(extract) && descriptors) &&
            !addressOnlyCompared(extract.getResult(), module, /*depth=*/6))
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
static bool isFreshUnaliased(Value v, ModuleOp module, FreshMemo &memo) {
  if (v.getDefiningOp<memref::AllocOp>())
    return true;
  auto arg = dyn_cast<BlockArgument>(v);
  auto func = arg ? dyn_cast<func::FuncOp>(arg.getOwner()->getParentOp()) : func::FuncOp();
  if (!func || arg.getOwner() != &func.getBody().front() || func.isPublic())
    return false;
  if (auto known = memo.answers.find(v); known != memo.answers.end())
    return known->second;
  memo.answers[v] = false;
  if (!memo.users) {
    memo.tables = std::make_unique<SymbolTableCollection>();
    memo.users = std::make_unique<SymbolUserMap>(*memo.tables, module);
  }
  bool fresh = true;
  for (Operation *user : memo.users->getUsers(func)) {
    auto call = dyn_cast<func::CallOp>(user);
    if (!call) {
      if (const char *d = std::getenv("CLEAVE_DEBUG_PARALLELIZE"); d && std::string(d) == "1")
        llvm::errs() << "  not fresh: " << func.getName() << " used by " << user->getName() << '\n';
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
  memo.answers[v] = fresh;
  return fresh;
}

// A value an inner loop of `cleaveCooperativeRows` carries as a reduction:
// `acc = acc <op> x` for `combine` an associative and commutative `arith` op,
// or the conditional update `acc = compare(x, acc) ? x : acc` (a running
// maximum or minimum: an `arith.select`, or cleave's `if s > m { m = s; }`,
// an `scf.if` whose branch reads `x` again). `x` doesn't depend on `acc`.
struct CarriedReduction {
  unsigned index;
  Value x;
  Operation *combine = nullptr; // the binary form
  Operation *compare = nullptr; // the conditional form
  // The ops replaced by the combination, to leave out of the per-thread loop.
  SmallVector<Operation *> chain;
  // `combine(u, v)`, `u` in the accumulator's place, `v` in `x`'s.
  Value apply(OpBuilder &b, Location loc, Value acc, Value u, Value v) const {
    IRMapping mapping;
    mapping.map(acc, u);
    mapping.map(x, v);
    if (combine)
      return b.clone(*combine, mapping)->getResult(0);
    Value taken = b.clone(*compare, mapping)->getResult(0);
    return arith::SelectOp::create(b, loc, taken, v, u);
  }
};

// Whether two values are loads of the same element: the same memref, the
// same indices up to casts (cleave's `if s[i, j] > m { m = s[i, j]; }` casts
// `i` and `j` again in the branch).
static bool sameLoad(Value a, Value b) {
  auto la = a.getDefiningOp<memref::LoadOp>(), lb = b.getDefiningOp<memref::LoadOp>();
  if (!la || !lb || la.getMemRef() != lb.getMemRef() || la.getIndices().size() != lb.getIndices().size())
    return false;
  auto uncast = [](Value v) {
    while (auto cast = v.getDefiningOp<arith::IndexCastOp>())
      v = cast.getIn();
    return v;
  };
  for (auto [x, y] : llvm::zip(la.getIndices(), lb.getIndices()))
    if (uncast(x) != uncast(y))
      return false;
  return true;
}

// The reductions `loop` carries, one per iter_arg, if every one is of a form
// `CarriedReduction` knows and the loop writes no memory; none otherwise.
static std::optional<SmallVector<CarriedReduction>> carriedReductions(scf::ForOp loop) {
  Block *body = loop.getBody();
  bool writes = false;
  loop.getBody()->walk([&](Operation *inner) {
    if (!isMemoryEffectFree(inner) && !isa<memref::LoadOp>(inner) &&
        !inner->hasTrait<OpTrait::HasRecursiveMemoryEffects>())
      writes = true;
  });
  if (writes)
    return std::nullopt;
  auto yield = cast<scf::YieldOp>(body->getTerminator());
  SmallVector<CarriedReduction> reductions;
  for (auto [index, acc] : llvm::enumerate(loop.getRegionIterArgs())) {
    if (!acc.getType().isIntOrFloat() || acc.getType().getIntOrFloatBitWidth() != 32)
      return std::nullopt;
    CarriedReduction r;
    r.index = index;
    Operation *def = yield.getOperand(index).getDefiningOp();
    if (!def || def->getBlock() != body)
      return std::nullopt;
    if (isa<arith::AddFOp, arith::MulFOp, arith::MaximumFOp, arith::MinimumFOp, arith::MaxNumFOp, arith::MinNumFOp,
            arith::AddIOp, arith::MulIOp, arith::MaxSIOp, arith::MinSIOp, arith::MaxUIOp, arith::MinUIOp,
            arith::AndIOp, arith::OrIOp, arith::XOrIOp>(def)) {
      Value l = def->getOperand(0), rhs = def->getOperand(1);
      if (l != acc && rhs != acc)
        return std::nullopt;
      r.x = l == acc ? rhs : l;
      r.combine = def;
      r.chain = {def};
    } else {
      // The conditional form: `select(c, x, acc)` or `scf.if c { x } else { acc }`,
      // `c` comparing `x` and `acc`.
      Value condition, taken;
      if (auto select = dyn_cast<arith::SelectOp>(def)) {
        if (select.getFalseValue() != acc)
          return std::nullopt;
        condition = select.getCondition(), taken = select.getTrueValue();
      } else if (auto branch = dyn_cast<scf::IfOp>(def); branch && branch.getNumResults() == 1 &&
                                                         branch.elseBlock()) {
        auto thenYield = cast<scf::YieldOp>(branch.thenBlock()->getTerminator());
        auto elseYield = cast<scf::YieldOp>(branch.elseBlock()->getTerminator());
        if (elseYield.getOperand(0) != acc || branch.elseBlock()->getOperations().size() != 1)
          return std::nullopt;
        // The branch only reads (again) what it yields.
        if (llvm::any_of(branch.thenBlock()->without_terminator(),
                         [](Operation &inner) { return !isa<memref::LoadOp>(inner) && !isPure(&inner); }))
          return std::nullopt;
        condition = branch.getCondition(), taken = thenYield.getOperand(0);
      } else {
        return std::nullopt;
      }
      Operation *compare = condition.getDefiningOp();
      if (!compare || !isa<arith::CmpFOp, arith::CmpIOp>(compare) || compare->getBlock() != body)
        return std::nullopt;
      Value a = compare->getOperand(0), b = compare->getOperand(1);
      Value other = a == acc ? b : b == acc ? a : Value();
      if (!other || !(taken == other || sameLoad(taken, other)))
        return std::nullopt;
      r.x = other;
      r.compare = compare;
      r.chain = {compare, def};
    }
    // `acc` used by its combination only, which nothing else uses.
    for (Operation *user : acc.getUsers())
      if (!llvm::is_contained(r.chain, user) && !llvm::any_of(r.chain, [&](Operation *c) {
            return c->isAncestor(user);
          }))
        return std::nullopt;
    for (Operation *op : r.chain)
      for (Operation *user : op->getUsers())
        if (user != yield && !llvm::is_contained(r.chain, user))
          return std::nullopt;
    reductions.push_back(std::move(r));
  }
  // No `x` depends on a carried value.
  llvm::DenseSet<Value> carried(loop.getRegionIterArgs().begin(), loop.getRegionIterArgs().end());
  for (Operation &op : body->without_terminator()) {
    if (llvm::any_of(reductions, [&](const CarriedReduction &r) { return llvm::is_contained(r.chain, &op); }))
      continue;
    bool depends = false;
    op.walk([&](Operation *inner) {
      for (Value operand : inner->getOperands())
        depends |= carried.contains(operand);
    });
    if (depends)
      return std::nullopt;
  }
  return reductions;
}

// Whether the iterations of the inner loop `loop` may run on different
// threads: it calls nothing, declares the effects of everything, and every
// memref it writes it reaches only with its induction variable at one same
// index position, so that iterations touch disjoint elements of it.
static bool elementsAreIndependent(scf::ForOp loop) {
  if (loop.getNumRegionIterArgs() != 0)
    return false;
  Value iv = loop.getInductionVar();
  llvm::DenseSet<Value> written;
  bool ok = true;
  loop.getBody()->walk([&](Operation *inner) {
    if (isa<CallOpInterface>(inner)) {
      ok = false;
      return;
    }
    if (auto store = dyn_cast<memref::StoreOp>(inner))
      written.insert(viewRoot(store.getMemRef()));
    else if (!isa<memref::LoadOp>(inner) && !isMemoryEffectFree(inner) &&
             !inner->hasTrait<OpTrait::HasRecursiveMemoryEffects>())
      ok = false;
  });
  if (!ok || written.empty())
    return false;
  llvm::DenseMap<Value, unsigned> position;
  loop.getBody()->walk([&](Operation *inner) {
    Value accessed;
    ValueRange indices;
    if (auto load = dyn_cast<memref::LoadOp>(inner))
      accessed = load.getMemRef(), indices = load.getIndices();
    else if (auto store = dyn_cast<memref::StoreOp>(inner))
      accessed = store.getMemRef(), indices = store.getIndices();
    if (!accessed || !written.contains(viewRoot(accessed)))
      return;
    if (accessed != viewRoot(accessed)) {
      ok = false; // through a view: positions don't compare
      return;
    }
    std::optional<unsigned> here;
    for (auto [i, index] : llvm::enumerate(indices))
      if (isInductionVariable(index, iv))
        here = i;
    auto [it, fresh] = position.try_emplace(accessed, here.value_or(~0u));
    if (!here || (!fresh && it->second != *here))
      ok = false;
  });
  return ok;
}

// Vulkan backend: a parallel loop over rows whose body runs inner loops over
// the row (`causal_softmax`: a maximum, exponentials, a sum, a scaling;
// `rms_norm`, `log_sum_exp_row`, a `linalg` reduction lowered) made a
// `gpu.launch` of a workgroup of `threads` per row instead of a thread per
// row, which left the device all but idle (nanoLM's softmax: 16 workgroups,
// 43% of a step's device time). In the workgroup, every thread computes the
// body's scalars; each inner loop whose elements are independent
// (`elementsAreIndependent`) is shared out, element `j` to thread `j mod
// threads`, a barrier after it; each inner loop carrying reductions
// (`carriedReductions`) gives every thread a partial value over its share,
// combined by a tree in workgroup memory of fixed shape, the same result
// every run, though summed in another order than a single thread did; a
// store of the body itself is thread 0's. Barriers order device memory too:
// one loop's elements are other threads' in the next. A body with anything
// else stays a thread per row. Returns how many loops it rewrote.
static int64_t cleaveCooperativeRows(MlirOperation op, int64_t threads) {
  auto module = dyn_cast<ModuleOp>(unwrap(op));
  if (!module)
    return 0;
  MLIRContext *ctx = module.getContext();
  SmallVector<scf::ParallelOp> candidates;
  module.walk([&](scf::ParallelOp loop) {
    if (loop.getNumLoops() != 1 || loop.getNumResults() != 0 || loop->getParentOfType<gpu::GPUModuleOp>() ||
        loop->getParentOfType<gpu::LaunchOp>() || loop->getParentOfType<scf::ParallelOp>())
      return;
    Block *body = loop.getBody();
    bool hasInner = false, ok = true;
    static const bool debug = [] {
      const char *v = std::getenv("CLEAVE_DEBUG_PARALLELIZE");
      return v && std::string(v) == "1";
    }();
    for (Operation &inner : body->without_terminator()) {
      bool fits = true;
      if (auto forOp = dyn_cast<scf::ForOp>(inner)) {
        hasInner = true;
        fits = forOp.getNumRegionIterArgs() == 0 ? elementsAreIndependent(forOp) : bool(carriedReductions(forOp));
      } else if (!isa<memref::LoadOp, memref::StoreOp>(inner) &&
                 (inner.getNumRegions() != 0 || !isPure(&inner))) {
        fits = false;
      }
      if (!fits && ok && debug)
        llvm::errs() << "cleave-cooperative-rows: a row loop in " << loop->getParentOfType<func::FuncOp>().getName()
                     << " stays a thread per row: " << inner.getName() << '\n';
      ok &= fits;
    }
    if (ok && hasInner)
      candidates.push_back(loop);
  });
  auto barrier = [&](OpBuilder &b, Location loc) {
    spirv::ControlBarrierOp::create(
        b, loc, spirv::ScopeAttr::get(ctx, spirv::Scope::Workgroup), spirv::ScopeAttr::get(ctx, spirv::Scope::Workgroup),
        spirv::MemorySemanticsAttr::get(ctx, spirv::MemorySemantics::AcquireRelease |
                                                 spirv::MemorySemantics::WorkgroupMemory |
                                                 spirv::MemorySemantics::UniformMemory));
  };
  Attribute workgroup = spirv::StorageClassAttr::get(ctx, spirv::StorageClass::Workgroup);
  for (scf::ParallelOp loop : candidates) {
    OpBuilder builder(loop);
    Location loc = loop.getLoc();
    auto constIndex = [&](OpBuilder &b, int64_t v) -> Value { return arith::ConstantIndexOp::create(b, loc, v); };
    Value lower = loop.getLowerBound()[0], upper = loop.getUpperBound()[0], step = loop.getStep()[0];
    // Rows: ceil((upper - lower) / step).
    Value span = arith::SubIOp::create(builder, loc, upper, lower);
    Value rows = arith::CeilDivSIOp::create(builder, loc, span, step);
    Value one = constIndex(builder, 1);
    auto launch = gpu::LaunchOp::create(builder, loc, rows, one, one, constIndex(builder, threads), one, one);
    Block &kernel = launch.getBody().front();
    if (kernel.empty() || !kernel.back().hasTrait<OpTrait::IsTerminator>()) {
      OpBuilder atEnd = OpBuilder::atBlockEnd(&kernel);
      gpu::TerminatorOp::create(atEnd, loc);
    }
    OpBuilder kb(&kernel, std::prev(kernel.end()));
    Value tid = launch.getThreadIds().x;
    IRMapping mapping;
    mapping.map(loop.getInductionVars()[0],
                arith::AddIOp::create(kb, loc, lower,
                                      arith::MulIOp::create(kb, loc, launch.getBlockIds().x, step)).getResult());
    Value isFirst = arith::CmpIOp::create(kb, loc, arith::CmpIPredicate::eq, tid, constIndex(kb, 0));
    // An inner loop's bounds for this thread: from `lower + tid * step`, by
    // `threads * step`, in the loop's own integer type.
    auto share = [&](scf::ForOp inner) {
      Value lb = mapping.lookupOrDefault(inner.getLowerBound()), ub = mapping.lookupOrDefault(inner.getUpperBound()),
            st = mapping.lookupOrDefault(inner.getStep());
      Type type = lb.getType();
      Value t = type.isIndex() ? tid : arith::IndexCastOp::create(kb, loc, type, tid).getResult();
      Value n = type.isIndex() ? constIndex(kb, threads)
                               : arith::ConstantIntOp::create(kb, loc, type, threads).getResult();
      Value first = arith::AddIOp::create(kb, loc, lb, arith::MulIOp::create(kb, loc, t, st));
      return std::make_tuple(first, ub, Value(arith::MulIOp::create(kb, loc, st, n)));
    };
    for (Operation &inner : loop.getBody()->without_terminator()) {
      auto forOp = dyn_cast<scf::ForOp>(inner);
      if (auto store = dyn_cast<memref::StoreOp>(inner)) {
        auto only = scf::IfOp::create(kb, loc, TypeRange{}, isFirst, /*withElseRegion=*/false);
        OpBuilder thenBuilder = OpBuilder::atBlockBegin(only.thenBlock());
        thenBuilder.clone(inner, mapping);
        barrier(kb, loc);
      } else if (!forOp) {
        kb.clone(inner, mapping);
      } else if (forOp.getNumRegionIterArgs() == 0) {
        auto [first, ub, st] = share(forOp);
        scf::ForOp::create(kb, loc, first, ub, st, ValueRange{},
                           [&](OpBuilder &lb, Location, Value j, ValueRange) {
                             IRMapping inMapping = mapping;
                             inMapping.map(forOp.getInductionVar(), j);
                             for (Operation &op : forOp.getBody()->without_terminator())
                               lb.clone(op, inMapping);
                             scf::YieldOp::create(lb, loc);
                           });
        barrier(kb, loc);
      } else {
        SmallVector<CarriedReduction> reductions = *carriedReductions(forOp);
        auto [first, ub, st] = share(forOp);
        // Each thread's partials, and whether its share was empty.
        SmallVector<Value> inits;
        for (Value init : forOp.getInitArgs())
          inits.push_back(mapping.lookupOrDefault(init));
        SmallVector<Value> start(inits);
        Value none = arith::ConstantIntOp::create(kb, loc, kb.getI1Type(), 0);
        start.push_back(none);
        auto partial = scf::ForOp::create(
            kb, loc, first, ub, st, start, [&](OpBuilder &lb, Location, Value j, ValueRange acc) {
              IRMapping inMapping = mapping;
              inMapping.map(forOp.getInductionVar(), j);
              for (auto [arg, value] : llvm::zip(forOp.getRegionIterArgs(), acc.drop_back()))
                inMapping.map(arg, value);
              for (Operation &op : forOp.getBody()->without_terminator())
                if (!llvm::any_of(reductions, [&](const CarriedReduction &r) { return llvm::is_contained(r.chain, &op); }))
                  lb.clone(op, inMapping);
              Value seen = acc.back();
              SmallVector<Value> next;
              for (const CarriedReduction &r : reductions) {
                Value x = inMapping.lookupOrDefault(r.x), mine = acc[r.index];
                Value combined = r.apply(lb, loc, forOp.getRegionIterArgs()[r.index], mine, x);
                next.push_back(arith::SelectOp::create(lb, loc, seen, combined, x));
              }
              next.push_back(arith::ConstantIntOp::create(lb, loc, lb.getI1Type(), 1));
              scf::YieldOp::create(lb, loc, next);
            });
        // The tree: each partial and its flag in workgroup memory, halves
        // combined into the first until one is left.
        Value seen = partial.getResults().back();
        auto flagType = MemRefType::get({threads}, kb.getI32Type(), MemRefLayoutAttrInterface{}, workgroup);
        Value flags = memref::AllocOp::create(kb, loc, flagType);
        memref::StoreOp::create(kb, loc, arith::ExtUIOp::create(kb, loc, kb.getI32Type(), seen), flags, tid);
        SmallVector<Value> slots;
        for (const CarriedReduction &r : reductions) {
          Type type = forOp.getRegionIterArgs()[r.index].getType();
          Value slot = memref::AllocOp::create(
              kb, loc, MemRefType::get({threads}, type, MemRefLayoutAttrInterface{}, workgroup));
          memref::StoreOp::create(kb, loc, partial.getResult(r.index), slot, tid);
          slots.push_back(slot);
        }
        barrier(kb, loc);
        Value zero32 = arith::ConstantIntOp::create(kb, loc, kb.getI32Type(), 0);
        for (int64_t half = threads / 2; half >= 1; half /= 2) {
          Value active = arith::CmpIOp::create(kb, loc, arith::CmpIPredicate::ult, tid, constIndex(kb, half));
          auto round = scf::IfOp::create(kb, loc, TypeRange{}, active, /*withElseRegion=*/false);
          OpBuilder sb = OpBuilder::atBlockBegin(round.thenBlock());
          Value other = arith::AddIOp::create(sb, loc, tid, constIndex(sb, half));
          Value mineSeen = arith::CmpIOp::create(sb, loc, arith::CmpIPredicate::ne,
                                                 memref::LoadOp::create(sb, loc, flags, tid), zero32);
          Value otherSeen = arith::CmpIOp::create(sb, loc, arith::CmpIPredicate::ne,
                                                  memref::LoadOp::create(sb, loc, flags, other), zero32);
          for (auto [r, slot] : llvm::zip(reductions, slots)) {
            Value mine = memref::LoadOp::create(sb, loc, slot, tid);
            Value theirs = memref::LoadOp::create(sb, loc, slot, other);
            Value both = r.apply(sb, loc, forOp.getRegionIterArgs()[r.index], mine, theirs);
            Value merged = arith::SelectOp::create(sb, loc, mineSeen, both, theirs);
            memref::StoreOp::create(sb, loc, arith::SelectOp::create(sb, loc, otherSeen, merged, mine), slot, tid);
          }
          Value either = arith::OrIOp::create(sb, loc, mineSeen, otherSeen);
          memref::StoreOp::create(sb, loc, arith::ExtUIOp::create(sb, loc, sb.getI32Type(), either), flags, tid);
          barrier(kb, loc);
        }
        // The loop's results, in every thread: its initial values combined
        // with the total, when any thread had elements.
        Value any = arith::CmpIOp::create(kb, loc, arith::CmpIPredicate::ne,
                                          memref::LoadOp::create(kb, loc, flags, constIndex(kb, 0)), zero32);
        for (auto [r, slot] : llvm::zip(reductions, slots)) {
          Value total = memref::LoadOp::create(kb, loc, slot, constIndex(kb, 0));
          Value withInit = r.apply(kb, loc, forOp.getRegionIterArgs()[r.index], inits[r.index], total);
          mapping.map(forOp.getResult(r.index), arith::SelectOp::create(kb, loc, any, withInit, inits[r.index]).getResult());
        }
        // Every thread has read the totals before a later tree reuses no
        // buffer, but the next loop may read what this one's threads wrote.
        barrier(kb, loc);
      }
    }
    loop->erase();
  }
  return candidates.size();
}

static int64_t cleaveCooperativeRowsOf128(MlirOperation op) { return cleaveCooperativeRows(op, 128); }

// A tiling of a product for `cleaveTileMatmulsForGpu`: a workgroup computes
// a `bm` x `bn` tile of the result, each of its threads a `tm` x `tn` block
// of it (rows `ty + i * bm / tm`, columns `tx + j * bn / tn`: neighbouring
// threads on neighbouring columns), over slices of `bk` of the reduction
// staged in workgroup memory.
struct ProductTiling {
  int64_t bm, bn, bk, tm, tn;
};

// Vulkan backend: every `linalg` op that is a product (two parallel loops
// `m` and `n` indexing its output, one reduction `k`; one input indexed by
// `m` and `k`, the other by `k` and `n`, in either order: `matmul`, its
// transposed forms, `blas-to-linalg`'s `sgemm`) made a `gpu.launch` of its
// own, tiled for the device instead of the naive kernel (a thread per
// output element, every operand read from device memory: 1-2% of the
// FMA ceiling, `bench/gpu_matmul`). Each workgroup stages slices of its
// rows of A and columns of B in workgroup memory, loaded with neighbouring
// threads on neighbouring addresses of whichever dimension is contiguous;
// each thread keeps its block of the result in registers. Each element's
// value is the op's own body applied over `k` in order, as the naive
// kernel does: the same result to the bit, whatever the body (an `sgemm`'s
// `alpha` included). Statically shaped products only, whose sizes the
// tiling divides; the others stay on the naive path. Returns how many it
// tiled.
static int64_t cleaveTileMatmulsForGpu(MlirOperation op) {
  auto module = dyn_cast<ModuleOp>(unwrap(op));
  if (!module)
    return 0;
  MLIRContext *ctx = module.getContext();
  SmallVector<linalg::LinalgOp> products;
  module.walk([&](linalg::LinalgOp linalgOp) {
    if (linalgOp.hasPureBufferSemantics() && !linalgOp->getParentOfType<gpu::GPUModuleOp>() &&
        !linalgOp->getParentOfType<gpu::LaunchOp>() && linalgOp.getNumLoops() == 3 &&
        linalgOp.getNumDpsInputs() == 2 && linalgOp.getNumDpsInits() == 1 && linalgOp.getNumReductionLoops() == 1)
      products.push_back(linalgOp);
  });
  const ProductTiling tilings[] = {{128, 128, 8, 8, 8}, {64, 64, 16, 4, 4}};
  int64_t tiled = 0;
  for (linalg::LinalgOp product : products) {
    // The loops: `m` and `n` the output's dimensions, `k` the reduction.
    AffineMap outMap = product.getMatchingIndexingMap(product.getDpsInitOperand(0));
    if (outMap.getNumResults() != 2 || !outMap.isProjectedPermutation())
      continue;
    unsigned dimM = outMap.getDimPosition(0), dimN = outMap.getDimPosition(1);
    unsigned dimK = 3 - dimM - dimN;
    if (product.getIteratorTypesArray()[dimK] != utils::IteratorType::reduction)
      continue;
    // `a` the input indexed by `m` (and `k`), `b` the one by `n` (and `k`).
    OpOperand *a = product.getDpsInputOperand(0), *b = product.getDpsInputOperand(1);
    auto indexes = [&](OpOperand *operand, unsigned d1, unsigned d2) {
      AffineMap map = product.getMatchingIndexingMap(operand);
      return map.getNumResults() == 2 && map.isProjectedPermutation() &&
             ((map.getDimPosition(0) == d1 && map.getDimPosition(1) == d2) ||
              (map.getDimPosition(0) == d2 && map.getDimPosition(1) == d1));
    };
    if (!indexes(a, dimM, dimK))
      std::swap(a, b);
    if (!indexes(a, dimM, dimK) || !indexes(b, dimK, dimN))
      continue;
    Value c = product.getDpsInitOperand(0)->get();
    auto aType = dyn_cast<MemRefType>(a->get().getType()), bType = dyn_cast<MemRefType>(b->get().getType()),
         cType = dyn_cast<MemRefType>(c.getType());
    if (!aType || !bType || !cType || !aType.hasStaticShape() || !bType.hasStaticShape() || !cType.hasStaticShape())
      continue;
    // The body: operations without memory effects or regions, applied per
    // element in the kernel.
    Block *body = product.getBlock();
    if (llvm::any_of(body->without_terminator(), [](Operation &inner) {
          return inner.getNumRegions() != 0 || !isPure(&inner);
        }))
      continue;
    SmallVector<int64_t> ranges = product.getStaticLoopRanges();
    int64_t m = ranges[dimM], n = ranges[dimN], k = ranges[dimK];
    const ProductTiling *tiling = nullptr;
    for (const ProductTiling &candidate : tilings) {
      int64_t threads = (candidate.bm / candidate.tm) * (candidate.bn / candidate.tn);
      if (m % candidate.bm == 0 && n % candidate.bn == 0 && k % candidate.bk == 0 &&
          (candidate.bm * candidate.bk) % threads == 0 && (candidate.bn * candidate.bk) % threads == 0) {
        tiling = &candidate;
        break;
      }
    }
    if (!tiling)
      continue;
    const int64_t bm = tiling->bm, bn = tiling->bn, bk = tiling->bk, tm = tiling->tm, tn = tiling->tn;
    const int64_t threadsX = bn / tn, threadsY = bm / tm, threads = threadsX * threadsY;

    OpBuilder builder(product);
    Location loc = product.getLoc();
    auto constant = [&](OpBuilder &at, int64_t v) -> Value { return arith::ConstantIndexOp::create(at, loc, v); };
    // Workgroup memory, `[bk][bm + 1]` and `[bk][bn + 1]`: the padding puts
    // the column a thread stores into (a slice of a row-major A, `k` the
    // fastest) on a different bank from its neighbour's. Allocated in the
    // kernel, in SPIR-V's own storage class, the form `convert-gpu-to-spirv`
    // turns into workgroup variables: it handles neither a `gpu.func`'s
    // workgroup attributions nor `#gpu.address_space<workgroup>`.
    Attribute workgroup = spirv::StorageClassAttr::get(ctx, spirv::StorageClass::Workgroup);
    auto aShared = MemRefType::get({bk, bm + 1}, aType.getElementType(), MemRefLayoutAttrInterface{}, workgroup);
    auto bShared = MemRefType::get({bk, bn + 1}, bType.getElementType(), MemRefLayoutAttrInterface{}, workgroup);
    Value one = constant(builder, 1);
    auto launch = gpu::LaunchOp::create(builder, loc, constant(builder, n / bn), constant(builder, m / bm), one,
                                        constant(builder, threadsX), constant(builder, threadsY), one,
                                        /*dynamicSharedMemorySize=*/nullptr, /*asyncTokenType=*/nullptr);
    Block &kernel = launch.getBody().front();
    if (kernel.empty() || !kernel.back().hasTrait<OpTrait::IsTerminator>()) {
      OpBuilder atEnd = OpBuilder::atBlockEnd(&kernel);
      gpu::TerminatorOp::create(atEnd, loc);
    }
    OpBuilder kb(&kernel, std::prev(kernel.end()));
    Value tx = launch.getThreadIds().x, ty = launch.getThreadIds().y;
    Value as = memref::AllocOp::create(kb, loc, aShared), bs = memref::AllocOp::create(kb, loc, bShared);
    auto add = [&](OpBuilder &at, Value x, Value y) -> Value { return arith::AddIOp::create(at, loc, x, y); };
    auto mul = [&](OpBuilder &at, Value x, int64_t y) -> Value {
      return arith::MulIOp::create(at, loc, x, constant(at, y));
    };
    Value m0 = mul(kb, launch.getBlockIds().y, bm), n0 = mul(kb, launch.getBlockIds().x, bn);
    Value tid = add(kb, mul(kb, ty, threadsX), tx);
    // An operand's indices for loop values `at` (by loop dimension), in the
    // order its map reads them.
    auto indicesOf = [&](OpOperand *operand, ArrayRef<std::pair<unsigned, Value>> at) {
      AffineMap map = product.getMatchingIndexingMap(operand);
      SmallVector<Value> indices;
      for (unsigned r = 0; r < map.getNumResults(); ++r)
        for (auto [dim, value] : at)
          if (map.getDimPosition(r) == dim)
            indices.push_back(value);
      return indices;
    };
    // This thread's elements of the result: row `i`, column `j` of its block.
    auto rowOf = [&](OpBuilder &at, int64_t i) { return add(at, m0, add(at, ty, constant(at, i * threadsY))); };
    auto columnOf = [&](OpBuilder &at, int64_t j) { return add(at, n0, add(at, tx, constant(at, j * threadsX))); };
    SmallVector<Value> initial;
    for (int64_t i = 0; i < tm; ++i)
      for (int64_t j = 0; j < tn; ++j)
        initial.push_back(memref::LoadOp::create(
            kb, loc, c, indicesOf(product.getDpsInitOperand(0), {{dimM, rowOf(kb, i)}, {dimN, columnOf(kb, j)}})));
    // Stages a `rows` x `bk` slice of `operand` (its `outer` dimension
    // `rows` long from `origin`, `k` from `k0`) into `shared[k][row]`, every
    // thread a few elements, neighbouring threads on the operand's contiguous
    // dimension.
    auto stage = [&](OpBuilder &at, OpOperand *operand, Value shared, unsigned outer, Value origin, int64_t rows,
                     Value k0) {
      bool kFastest = product.getMatchingIndexingMap(operand).getDimPosition(1) == dimK;
      for (int64_t r = 0; r < rows * bk / threads; ++r) {
        Value l = add(at, tid, constant(at, r * threads));
        int64_t fast = kFastest ? bk : rows;
        Value low = arith::RemUIOp::create(at, loc, l, constant(at, fast));
        Value high = arith::DivUIOp::create(at, loc, l, constant(at, fast));
        Value kk = kFastest ? low : high, row = kFastest ? high : low;
        Value v = memref::LoadOp::create(at, loc, operand->get(),
                                         indicesOf(operand, {{outer, add(at, origin, row)}, {dimK, add(at, k0, kk)}}));
        memref::StoreOp::create(at, loc, v, shared, ValueRange{kk, row});
      }
    };
    Value aArg = body->getArgument(a->getOperandNumber()), bArg = body->getArgument(b->getOperandNumber()),
          cArg = body->getArgument(2);
    Value yielded = body->getTerminator()->getOperand(0);
    auto outerLoop = scf::ForOp::create(
        kb, loc, constant(kb, 0), constant(kb, k), constant(kb, bk), initial,
        [&](OpBuilder &lb, Location, Value k0, ValueRange accumulated) {
          stage(lb, a, as, dimM, m0, bm, k0);
          stage(lb, b, bs, dimN, n0, bn, k0);
          gpu::BarrierOp::create(lb, loc);
          auto innerLoop = scf::ForOp::create(
              lb, loc, constant(lb, 0), constant(lb, bk), constant(lb, 1), accumulated,
              [&](OpBuilder &ib, Location, Value kk, ValueRange acc) {
                SmallVector<Value> aValues, bValues, next;
                for (int64_t i = 0; i < tm; ++i)
                  aValues.push_back(memref::LoadOp::create(
                      ib, loc, as, ValueRange{kk, add(ib, ty, constant(ib, i * threadsY))}));
                for (int64_t j = 0; j < tn; ++j)
                  bValues.push_back(memref::LoadOp::create(
                      ib, loc, bs, ValueRange{kk, add(ib, tx, constant(ib, j * threadsX))}));
                for (int64_t i = 0; i < tm; ++i)
                  for (int64_t j = 0; j < tn; ++j) {
                    IRMapping mapping;
                    mapping.map(aArg, aValues[i]);
                    mapping.map(bArg, bValues[j]);
                    mapping.map(cArg, acc[i * tn + j]);
                    for (Operation &inner : body->without_terminator())
                      ib.clone(inner, mapping);
                    next.push_back(mapping.lookupOrDefault(yielded));
                  }
                scf::YieldOp::create(ib, loc, next);
              });
          gpu::BarrierOp::create(lb, loc);
          scf::YieldOp::create(lb, loc, innerLoop.getResults());
        });
    for (int64_t i = 0; i < tm; ++i)
      for (int64_t j = 0; j < tn; ++j)
        memref::StoreOp::create(
            kb, loc, outerLoop.getResult(i * tn + j), c,
            indicesOf(product.getDpsInitOperand(0), {{dimM, rowOf(kb, i)}, {dimN, columnOf(kb, j)}}));
    product->erase();
    ++tiled;
  }
  return tiled;
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
  FreshMemo fresh;
  SmallVector<std::pair<Operation *, Value>> overwrites;
  // Every operation that is or holds a kernel-to-be, found in one walk
  // (walking each fresh buffer's first use instead took nanoLM's compile
  // quadratic: a first use can be the whole training loop).
  llvm::DenseSet<Operation *> holdsKernel;
  module.walk([&](Operation *inner) {
    if (!isa<linalg::LinalgOp, scf::ParallelOp, gpu::LaunchOp>(inner))
      return;
    for (Operation *at = inner; at && !isa<func::FuncOp>(at); at = at->getParentOp())
      if (!holdsKernel.insert(at).second)
        break;
  });
  // `fresh` (undefined until written) in `block`: a discard before its first
  // use, if that use is or holds a kernel.
  auto discardBeforeFirstUse = [&](Value freshMemref, Block *block) {
    auto type = dyn_cast<MemRefType>(freshMemref.getType());
    if (!type || !type.hasStaticShape())
      return;
    // The first use, through views, as the operation of `block` that holds it.
    Operation *first = nullptr;
    SmallVector<Value> pending{freshMemref};
    while (!pending.empty()) {
      Value v = pending.pop_back_val();
      for (Operation *user : v.getUsers()) {
        // A view or a cast only names the memory again (`blas-to-linalg`'s
        // operands are casts to a strided layout): its users count.
        if (isa<ViewLikeOpInterface, memref::CastOp>(user)) {
          for (Value result : user->getResults())
            pending.push_back(result);
          continue;
        }
        Operation *holder = block->findAncestorOpInBlock(*user);
        if (holder && (!first || holder->isBeforeInBlock(first)))
          first = holder;
      }
    }
    if (!first)
      return;
    // A kernel there: an access through `first`'s own values counts (a
    // loop carrying the buffer, causal attention's `out`). The discard is
    // right whatever `first` is; this only spares a mirror nothing would use.
    bool kernel = holdsKernel.contains(first);
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

  // The kernels' modules and functions, by name, found once: they are
  // top-level operations, and a search of the whole module per launch made
  // this pass most of nanoLM's compile (50 s for 1875 launches).
  llvm::StringMap<spirv::ModuleOp> spirvModules;
  llvm::StringMap<gpu::GPUModuleOp> gpuModules;
  for (Operation &child : module.getBody()->getOperations()) {
    if (auto spirvModule = dyn_cast<spirv::ModuleOp>(child); spirvModule && spirvModule.getName())
      spirvModules[*spirvModule.getName()] = spirvModule;
    else if (auto gpuModule = dyn_cast<gpu::GPUModuleOp>(child))
      gpuModules[gpuModule.getName()] = gpuModule;
  }

  HeapMemo heapParams;
  int64_t lowered = 0;
  for (gpu::LaunchFuncOp launch : launches) {
    StringRef kernelModule = launch.getKernelModuleName().getValue();
    StringRef kernel = launch.getKernelName().getValue();
    // Serialized at its first launch only.
    std::string binaryName = ("__cleave_spirv_" + kernelModule).str();
    memref::GlobalOp binary = globals.lookup(binaryName);
    if (!binary) {
      spirv::ModuleOp spirvModule = spirvModules.lookup(("__spv__" + kernelModule).str());
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
      binary = constantGlobal(binaryName, DenseElementsAttr::get(wordsType, wordValues));
    }
    int64_t wordCount = binary.getType().getNumElements();
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
    gpu::GPUModuleOp gpuModule = gpuModules.lookup(kernelModule);
    auto kernelFunc = gpuModule ? gpuModule.lookupSymbol<gpu::GPUFuncOp>(kernel) : gpu::GPUFuncOp();
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
                         ValueRange{addressOf(binary), constI64(wordCount), addressOf(entry), asI64(grid.x),
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

namespace {

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
CLEAVE_COUNTING_PASS(TileMatmulsForGpuPass, "cleave-tile-matmuls-for-gpu",
                     "products made tiled GPU kernels, through workgroup memory",
                     cleaveTileMatmulsForGpu)
CLEAVE_COUNTING_PASS(CooperativeRowsPass, "cleave-cooperative-rows",
                     "row loops made a workgroup per row, inner loops shared among its threads",
                     cleaveCooperativeRowsOf128)
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

} // namespace

void registerGpuPasses() {
  PassRegistration<TileParallelLoopsForGpuPass>();
  PassRegistration<SetSpirvEntryAbiPass>();
  PassRegistration<ParallelIteratorsFirstPass>();
  PassRegistration<MarkFullOverwritesPass>();
  PassRegistration<GuardZeroTripLoopsPass>();
  PassRegistration<CooperativeRowsPass>();
  PassRegistration<TileMatmulsForGpuPass>();
  PassRegistration<SinkViewsIntoLaunchesPass>();
  PassRegistration<BlasToLinalgPass>();
  PassRegistration<ParallelizeLoopsPass>();
  PassRegistration<GpuCopiesPass>();
  PassRegistration<SyncHostAccessesPass>();
  PassRegistration<LowerGpuLaunchesPass>();
}
