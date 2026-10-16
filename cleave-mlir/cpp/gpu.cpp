// The Vulkan backend's passes (`doc/plan-vulkan.md`): loops made kernels,
// the transfers between host and device made explicit, kernels lowered to
// SPIR-V and to calls of the runtime (`cleave-vk`).

#include "cleave.h"
#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/Transforms/RegionUtils.h"
#include <deque>
#include <set>

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

// A term of an affine form (`Affine`): a value, by its opaque pointer, or a
// quotient or remainder by a constant (`AffineForms::Division`).
using Term = const void *;
static Term termOf(Value v) { return v.getAsOpaquePointer(); }

// `terms` (coefficients by term) plus `constant`.
struct Affine {
  llvm::SmallDenseMap<Term, int64_t, 4> terms;
  int64_t constant = 0;
};

// Integer values as affine forms of the variables of `scope` (the induction
// variables of its `scf.for`s and `scf.parallel`s, the block and thread ids
// of its `gpu.launch`es) and of the values defined outside it: through
// constants, `arith`'s additions, subtractions, multiplications by a
// constant and casts between integer types (an index computation in range
// doesn't wrap), `affine.apply`, and, with `divisions`, unsigned divisions
// and remainders by a positive constant (`arith.divui`, `arith.remui`,
// `floordiv`, `mod`), each a variable of its own its dividend constrains
// (`Division`). None for anything else computed in `scope` (a load, a
// signed division, a loop-carried value).
struct AffineForms {
  explicit AffineForms(Operation *scope, bool divisions = false) : scope(scope), divisions(divisions) {}
  Operation *scope;
  bool divisions;
  // `quotient` (or the remainder) of `dividend` by `divisor`: `divisor *
  // quotient <= dividend <= divisor * quotient + divisor - 1`, the remainder
  // `dividend - divisor * quotient`.
  struct Division {
    Affine dividend;
    int64_t divisor;
    bool remainder;
  };
  std::deque<Division> quotients; // their addresses are their terms
  llvm::DenseSet<Term> divisionTerms;
  llvm::DenseMap<Term, Value> values;

  // Whether `v`, a block argument, is a variable of `scope`.
  bool isVariable(Value v) const {
    auto argument = dyn_cast<BlockArgument>(v);
    if (!argument)
      return false;
    Operation *owner = argument.getOwner()->getParentOp();
    if (auto loop = dyn_cast<scf::ForOp>(owner))
      return v == loop.getInductionVar();
    if (isa<scf::ParallelOp>(owner))
      return true;
    if (auto launch = dyn_cast<gpu::LaunchOp>(owner))
      return argument.getOwner() == &launch.getBody().front() && argument.getArgNumber() < 6;
    return false;
  }
  // A `gpu.launch`'s grid or workgroup size argument: the value it was
  // launched with.
  static Value launchSize(Value v) {
    auto argument = dyn_cast<BlockArgument>(v);
    if (!argument)
      return {};
    auto launch = dyn_cast<gpu::LaunchOp>(argument.getOwner()->getParentOp());
    if (!launch || argument.getOwner() != &launch.getBody().front())
      return {};
    gpu::KernelDim3 grid = launch.getGridSize(), block = launch.getBlockSize();
    gpu::KernelDim3 gridOperands = launch.getGridSizeOperandValues(),
                    blockOperands = launch.getBlockSizeOperandValues();
    for (auto [arg, operand] : llvm::zip(SmallVector<Value>{grid.x, grid.y, grid.z, block.x, block.y, block.z},
                                         SmallVector<Value>{gridOperands.x, gridOperands.y, gridOperands.z,
                                                            blockOperands.x, blockOperands.y, blockOperands.z}))
      if (arg == v)
        return operand;
    return {};
  }
  // Whether `term` takes a value of its own in each iteration of `scope`:
  // a variable or what is computed from one, not a value defined outside.
  bool perIteration(Term term) const {
    if (divisionTerms.contains(term))
      return true;
    Value v = values.lookup(term);
    Operation *at = v.getDefiningOp() ? v.getDefiningOp() : v.getParentBlock()->getParentOp();
    return scope->isAncestor(at);
  }
  Affine single(Value v) {
    Affine result;
    values[termOf(v)] = v;
    result.terms[termOf(v)] = 1;
    return result;
  }
  static Affine sum(const Affine &a, const Affine &b, int64_t sign) {
    Affine result = a;
    for (auto [term, c] : b.terms)
      result.terms[term] += sign * c;
    result.constant += sign * b.constant;
    return result;
  }
  static std::optional<Affine> product(const Affine &a, const Affine &b) {
    if (!a.terms.empty() && !b.terms.empty())
      return std::nullopt;
    const Affine &scaled = a.terms.empty() ? b : a;
    int64_t factor = a.terms.empty() ? a.constant : b.constant;
    Affine result;
    for (auto [term, c] : scaled.terms)
      result.terms[term] = c * factor;
    result.constant = scaled.constant * factor;
    return result;
  }
  std::optional<Affine> divide(std::optional<Affine> dividend, std::optional<Affine> divisor, bool remainder) {
    if (!divisions || !dividend || !divisor || !divisor->terms.empty() || divisor->constant <= 0)
      return std::nullopt;
    if (dividend->terms.empty()) {
      Affine result;
      int64_t q = dividend->constant >= 0 ? dividend->constant / divisor->constant
                                          : -((-dividend->constant + divisor->constant - 1) / divisor->constant);
      result.constant = remainder ? dividend->constant - q * divisor->constant : q;
      return result;
    }
    quotients.push_back({*dividend, divisor->constant, remainder});
    Term term = &quotients.back();
    divisionTerms.insert(term);
    Affine result;
    result.terms[term] = 1;
    return result;
  }
  std::optional<Affine> of(AffineExpr expr, ValueRange operands, unsigned dims) {
    if (auto c = dyn_cast<AffineConstantExpr>(expr)) {
      Affine result;
      result.constant = c.getValue();
      return result;
    }
    if (auto d = dyn_cast<AffineDimExpr>(expr))
      return of(operands[d.getPosition()]);
    if (auto s = dyn_cast<AffineSymbolExpr>(expr))
      return of(operands[dims + s.getPosition()]);
    auto binary = cast<AffineBinaryOpExpr>(expr);
    std::optional<Affine> lhs = of(binary.getLHS(), operands, dims), rhs = of(binary.getRHS(), operands, dims);
    if (!lhs || !rhs)
      return std::nullopt;
    switch (expr.getKind()) {
    case AffineExprKind::Add:
      return sum(*lhs, *rhs, 1);
    case AffineExprKind::Mul:
      return product(*lhs, *rhs);
    case AffineExprKind::FloorDiv:
      return divide(lhs, rhs, false);
    case AffineExprKind::Mod:
      return divide(lhs, rhs, true);
    default:
      return std::nullopt;
    }
  }
  std::optional<Affine> of(Value v) {
    if (std::optional<int64_t> c = getConstantIntValue(v)) {
      Affine result;
      result.constant = *c;
      return result;
    }
    if (Value size = launchSize(v))
      return of(size);
    Operation *def = v.getDefiningOp();
    Operation *at = def ? def : cast<BlockArgument>(v).getOwner()->getParentOp();
    if (!scope->isAncestor(at) || (!def && isVariable(v)))
      return single(v);
    if (!def)
      return std::nullopt;
    if (isa<arith::IndexCastOp, arith::IndexCastUIOp, arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp>(def))
      return of(def->getOperand(0));
    if (auto apply = dyn_cast<affine::AffineApplyOp>(def))
      return of(apply.getAffineMap().getResult(0), apply.getMapOperands(), apply.getAffineMap().getNumDims());
    if (!isa<arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::DivUIOp, arith::RemUIOp>(def))
      return std::nullopt;
    std::optional<Affine> lhs = of(def->getOperand(0)), rhs = of(def->getOperand(1));
    if (!lhs || !rhs)
      return std::nullopt;
    if (isa<arith::DivUIOp, arith::RemUIOp>(def))
      return divide(lhs, rhs, isa<arith::RemUIOp>(def));
    if (isa<arith::MulIOp>(def))
      return product(*lhs, *rhs);
    return sum(*lhs, *rhs, isa<arith::SubIOp>(def) ? -1 : 1);
  }
};

// `affineOf(v, scope)`: `v` as an affine form of `scope`'s variables, no
// divisions (`AffineForms`).
static std::optional<Affine> affineOf(Value v, Operation *scope) {
  AffineForms forms(scope);
  return forms.of(v);
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
      // A dimension the last index moves by one along (`rope`'s `h * DH +
      // j`: `j`), or that it is the sum of with something else.
      Value last = indices.back();
      std::optional<Affine> affine = affineOf(last, loop);
      for (auto [d, iv] : llvm::enumerate(loop.getInductionVars())) {
        auto add = last.getDefiningOp<arith::AddIOp>();
        if (affine ? affine->terms.lookup(termOf(iv)) == 1 : last == iv || (add && (add.getLhs() == iv || add.getRhs() == iv)))
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
    // A dimension too short to fill a workgroup shares it with the innermost
    // other one (`rope`'s 32 pairs with its heads).
    if (threads % sizes[chosen] == 0 && sizes[chosen] < threads)
      for (size_t d = extents.size(); d-- > 0;)
        if (d != chosen && extents[d] > 1) {
          sizes[d] = std::min(threads / sizes[chosen], extents[d]);
          break;
        }
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
// The runtime's bfloat16 tensor-core product (`cleave-vk`, `cleave_vk_gemm_bf16`), and the marker of
// a `linalg.matmul` the program asked to compute in bf16 (`stdlib/linalg`'s `MatMulBf16`,
// `mlir_lower.rs`).
static constexpr const char *kGemmBf16 = "cleave_vk_gemm_bf16";
static constexpr const char *kGemmBf16Operands = "cleave_vk_gemm_bf16_operands";
static constexpr const char *kComputeAttr = "cleave.compute";

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

// A system of integer constraints over affine forms' terms (`AffineForms`),
// each term taken in one of two iterations of the forms' scope (1, 2; a
// remainder's quotient 3, 4; another term 5 and up) or, defined outside it,
// shared by both (0): rows `sum(c * term) + constant` `>= 0`
// (`inequalities`) or `== 0` (`equalities`). `isEmpty` first adds each
// variable's own constraints (`define`): a loop's bounds, a launch's sizes,
// a division's definition. What isn't affine isn't constrained: a bound left
// out only widens the system.
struct AffineSystem {
  using Key = std::pair<Term, int>;
  struct Row {
    llvm::SmallDenseMap<Key, int64_t, 8> coefficients;
    int64_t constant = 0;
  };
  AffineForms &forms;
  SmallVector<Row> inequalities, equalities;

  explicit AffineSystem(AffineForms &forms) : forms(forms) {}

  Key keyOf(Term term, int iteration) const { return {term, forms.perIteration(term) ? iteration : 0}; }
  void accumulate(Row &row, const Affine &affine, int iteration, int64_t sign) const {
    for (auto [term, c] : affine.terms)
      row.coefficients[keyOf(term, iteration)] += sign * c;
    row.constant += sign * affine.constant;
  }
  Row rowOf(const Affine &affine, int iteration) const {
    Row row;
    accumulate(row, affine, iteration, 1);
    return row;
  }
  static Row constantRow(int64_t c) {
    Row row;
    row.constant = c;
    return row;
  }
  // `low <= x <= high`, either left out.
  void between(const Row &x, std::optional<Row> low, std::optional<Row> high) {
    if (low) {
      Row &row = inequalities.emplace_back(x); // x - low >= 0
      for (auto &[key, c] : low->coefficients)
        row.coefficients[key] -= c;
      row.constant -= low->constant;
    }
    if (high) {
      Row &row = inequalities.emplace_back(*high); // high - x >= 0
      for (auto &[key, c] : x.coefficients)
        row.coefficients[key] -= c;
      row.constant -= x.constant;
    }
  }
  std::optional<Row> boundRow(Value bound, int iteration, int64_t shift) {
    std::optional<Affine> affine = forms.of(bound);
    if (!affine)
      return std::nullopt;
    Row row = rowOf(*affine, iteration);
    row.constant += shift;
    return row;
  }
  // The constraints of each variable the rows use, in its iteration, and of
  // those they use in turn.
  void define() {
    llvm::DenseSet<Key> defined;
    for (bool more = true; more;) {
      more = false;
      SmallVector<Key> keys;
      for (SmallVectorImpl<Row> *rows : {&inequalities, &equalities})
        for (const Row &row : *rows)
          for (auto &entry : row.coefficients)
            if (entry.first.second == 1 || entry.first.second == 2)
              keys.push_back(entry.first);
      for (Key key : keys) {
        if (!defined.insert(key).second)
          continue;
        more = true;
        auto [term, iteration] = key;
        Row x;
        x.coefficients[key] = 1;
        if (forms.divisionTerms.contains(term)) {
          const AffineForms::Division &division = *static_cast<const AffineForms::Division *>(term);
          Row dividend = rowOf(division.dividend, iteration);
          // The quotient `q` (the remainder's own, a key of its own): `d *
          // q <= dividend <= d * q + d - 1`.
          Key quotientKey = division.remainder ? Key{term, iteration + 2} : key;
          Row scaled;
          scaled.coefficients[quotientKey] = division.divisor;
          Row top = scaled;
          top.constant = division.divisor - 1;
          between(dividend, scaled, top);
          if (division.remainder) {
            Row &row = equalities.emplace_back(x); // r - dividend + d * q == 0
            for (auto &[k, c] : dividend.coefficients)
              row.coefficients[k] -= c;
            row.constant -= dividend.constant;
            row.coefficients[quotientKey] += division.divisor;
          }
          continue;
        }
        Value v = forms.values.lookup(term);
        auto argument = dyn_cast<BlockArgument>(v);
        if (!argument)
          continue;
        Operation *owner = argument.getOwner()->getParentOp();
        if (auto loop = dyn_cast<scf::ForOp>(owner)) {
          between(x, boundRow(loop.getLowerBound(), iteration, 0), boundRow(loop.getUpperBound(), iteration, -1));
        } else if (auto loop = dyn_cast<scf::ParallelOp>(owner)) {
          unsigned d = argument.getArgNumber();
          between(x, boundRow(loop.getLowerBound()[d], iteration, 0),
                  boundRow(loop.getUpperBound()[d], iteration, -1));
        } else if (auto launch = dyn_cast<gpu::LaunchOp>(owner)) {
          gpu::KernelDim3 grid = launch.getGridSizeOperandValues(), block = launch.getBlockSizeOperandValues();
          Value sizes[] = {grid.x, grid.y, grid.z, block.x, block.y, block.z};
          between(x, constantRow(0), boundRow(sizes[argument.getArgNumber()], iteration, -1));
        }
      }
    }
  }
  // Whether no integer point satisfies the system (Presburger arithmetic,
  // exact), its variables' constraints added.
  bool isEmpty() {
    define();
    llvm::DenseMap<Key, unsigned> columns;
    for (SmallVectorImpl<Row> *set : {&inequalities, &equalities})
      for (const Row &row : *set)
        for (auto &entry : row.coefficients)
          columns.try_emplace(entry.first, columns.size());
    auto dense = [&](const Row &row) {
      SmallVector<int64_t> values(columns.size() + 1, 0);
      for (auto &entry : row.coefficients)
        values[columns.lookup(entry.first)] += entry.second;
      values.back() = row.constant;
      return values;
    };
    presburger::IntegerPolyhedron system(presburger::PresburgerSpace::getSetSpace(columns.size()));
    for (const Row &row : inequalities)
      system.addInequality(dense(row));
    for (const Row &row : equalities)
      system.addEquality(dense(row));
    return system.isIntegerEmpty();
  }
};

// Whether an access at `indices` (a `memref.load`'s or `memref.store`'s in
// `scope`) may reach the box of elements `box` (each index's lowest and
// highest): false only when no iteration of the loops and kernels around it
// can (`AffineSystem`); true too when an index isn't affine.
static bool mayReach(Operation *scope, ValueRange indices, ArrayRef<std::pair<int64_t, int64_t>> box) {
  AffineForms forms(scope, /*divisions=*/true);
  AffineSystem system(forms);
  for (auto [index, range] : llvm::zip(indices, box)) {
    std::optional<Affine> affine = forms.of(index);
    if (!affine)
      return true;
    system.between(system.rowOf(*affine, 1), AffineSystem::constantRow(range.first),
                   AffineSystem::constantRow(range.second));
  }
  return !system.isEmpty();
}

// Whether no element of `memref` one iteration of `scope` writes is read or
// written by another: an affine dependence test. `scope`'s iterations are
// the values of `band` (variables of `scope`: the induction variables of a
// nest of `scf.for`s, a host loop's launching kernels); everything else
// inside it (inner loops, a kernel's block and thread ids) ranges over what
// it ranges over in each. The accesses of `memref` in `scope`
// (`memref.load`, `memref.store`) grouped by the variable part of their
// indices, each index's constant part one of the group's: exactly when they
// make a progression (`c + s * t`, `0 <= t < n`: the 64 stores of a tiled
// product's block of results, `rope`'s two halves), else within their range;
// for each two groups, one writing, a system over two iterations, their own
// variables and the values defined outside `scope` they share
// (`AffineSystem`): the indices equal, the two iterations different along
// one variable of `band`. Each system has no integer solution or `memref` is
// shared. An index that isn't affine refuses, as do more groups than the
// test's systems are worth.
static bool accessesAreDisjoint(Operation *scope, ArrayRef<Value> band, Value memref) {
  AffineForms forms(scope, /*divisions=*/true);
  for (Value v : band)
    forms.single(v);
  struct Group {
    bool writes;
    SmallVector<Affine> indices; // their variable parts
    SmallVector<std::set<int64_t>> constants; // by index
  };
  // `std::deque`: a group's address (its constants') is a term.
  std::deque<Group> groups;
  bool ok = true;
  scope->walk([&](Operation *inner) {
    ValueRange indices;
    bool writes = false;
    if (auto load = dyn_cast<memref::LoadOp>(inner); load && load.getMemRef() == memref)
      indices = load.getIndices();
    else if (auto store = dyn_cast<memref::StoreOp>(inner); store && store.getMemRef() == memref)
      indices = store.getIndices(), writes = true;
    else
      return WalkResult::advance();
    SmallVector<Affine> variable;
    SmallVector<int64_t> constants;
    for (Value index : indices) {
      std::optional<Affine> affine = forms.of(index);
      if (!affine) {
        ok = false;
        return WalkResult::interrupt();
      }
      Affine part;
      for (auto [term, c] : affine->terms)
        if (c != 0)
          part.terms[term] = c;
      variable.push_back(part);
      constants.push_back(affine->constant);
    }
    auto same = [&](const Group &group) {
      return group.writes == writes && llvm::all_of(llvm::zip(group.indices, variable), [](auto pair) {
               auto &[x, y] = pair;
               return x.terms.size() == y.terms.size() &&
                      llvm::all_of(x.terms, [&](auto entry) { return y.terms.lookup(entry.first) == entry.second; });
             });
    };
    auto group = llvm::find_if(groups, same);
    if (group == groups.end()) {
      groups.push_back({writes, variable, SmallVector<std::set<int64_t>>(constants.size())});
      group = std::prev(groups.end());
    }
    for (auto [set, c] : llvm::zip(group->constants, constants))
      set.insert(c);
    return WalkResult::advance();
  });
  constexpr size_t kMostGroups = 64;
  if (!ok || groups.size() > kMostGroups)
    return false;
  using Key = AffineSystem::Key;
  using Row = AffineSystem::Row;
  for (size_t i = 0; i < groups.size(); ++i)
    for (size_t j = i; j < groups.size(); ++j) {
      const Group &a = groups[i], &b = groups[j];
      if (!a.writes && !b.writes)
        continue;
      AffineSystem base(forms);
      for (size_t k = 0; k < a.indices.size(); ++k) {
        // `a`'s index in the first iteration, `b`'s in the second, the same.
        Row row;
        base.accumulate(row, a.indices[k], 1, 1);
        base.accumulate(row, b.indices[k], 2, -1);
        for (auto [group, iteration, sign] : {std::tuple{&a, 1, int64_t(1)}, std::tuple{&b, 2, int64_t(-1)}}) {
          const std::set<int64_t> &constants = group->constants[k];
          int64_t lowest = *constants.begin(), highest = *constants.rbegin();
          row.constant += sign * lowest;
          if (constants.size() == 1)
            continue;
          // `lowest + step * t`, `0 <= t <= last`: a progression's terms, or
          // anything in the range (`step` 1).
          int64_t step = (highest - lowest) / int64_t(constants.size() - 1);
          bool progression = step > 0 && (highest - lowest) % int64_t(constants.size() - 1) == 0 &&
                             llvm::all_of(llvm::enumerate(constants), [&](auto entry) {
                               return entry.value() == lowest + step * int64_t(entry.index());
                             });
          if (!progression)
            step = 1;
          Key t{&constants, iteration + 4};
          row.coefficients[t] += sign * step;
          Row x;
          x.coefficients[t] = 1;
          base.between(x, AffineSystem::constantRow(0), AffineSystem::constantRow((highest - lowest) / step));
        }
        base.equalities.push_back(row);
      }
      for (Value dimension : band)
        for (int64_t direction : {1, -1}) {
          // The iterations differ along `dimension`: one's value before the
          // other's (`direction * (second - first) - 1 >= 0`).
          AffineSystem system = base;
          Row &apart = system.inequalities.emplace_back();
          apart.coefficients[{termOf(dimension), 2}] += direction;
          apart.coefficients[{termOf(dimension), 1}] -= direction;
          apart.constant = -1;
          if (!system.isEmpty())
            return false;
        }
    }
  return true;
}

// The memrefs the iterations of `loop` (an `scf.for` carrying nothing)
// write that they don't define, if it can be told which they are and that
// no one else can see them: its body calls nothing and declares the effects
// of every operation; every such memref is a local allocation
// (`allocationStaysLocal`), or an out-parameter the callers allocate fresh
// (`isFreshUnaliased`), that nothing in the function lets escape. What it
// only reads may be anything. `refuse` says why not.
static std::optional<SmallVector<Value>> privateWrites(scf::ForOp loop, ModuleOp module, FreshMemo &fresh,
                                                       llvm::function_ref<bool(const char *)> refuse) {
  if (loop.getNumRegionIterArgs() != 0)
    return refuse("carries values"), std::nullopt;
  for (Value bound : {loop.getLowerBound(), loop.getUpperBound(), loop.getStep()})
    if (Operation *def = bound.getDefiningOp(); def && loop->isAncestor(def))
      return refuse("bounds defined in the loop"), std::nullopt;
  llvm::SetVector<Value> written;
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
    return refuse("a call or an operation with undeclared effects"), std::nullopt;
  if (written.empty())
    return refuse("writes nothing"), std::nullopt;
  SmallVector<Value> shared;
  for (Value memref : written) {
    // Defined in the loop: private to an iteration.
    if (Operation *def = memref.getDefiningOp(); def && loop->isAncestor(def))
      continue;
    if (!isFreshUnaliased(memref, module, fresh))
      return refuse("writes a buffer that isn't a fresh allocation"), std::nullopt;
    if (!allocationStaysLocal(memref))
      return refuse("writes a buffer that escapes"), std::nullopt;
    shared.push_back(memref);
  }
  return shared;
}

// `CLEAVE_DEBUG_PARALLELIZE=1`: why a loop stays sequential, on stderr.
static bool debuggingParallelization() {
  static const bool debug = [] {
    const char *v = std::getenv("CLEAVE_DEBUG_PARALLELIZE");
    return v && std::string(v) == "1";
  }();
  return debug;
}

// Whether the iterations of `loop` (an `scf.for` carrying nothing) are
// independent: what it writes is its own (`privateWrites`), and each
// iteration touches its own elements of it: always the induction variable at
// one same index position (each iteration its own slice), or else no two
// iterations touching an element one writes (`accessesAreDisjoint`).
static bool iterationsAreIndependent(scf::ForOp loop, ModuleOp module, FreshMemo &fresh) {
  auto refuse = [&](const char *why) {
    if (debuggingParallelization()) {
      auto func = loop->getParentOfType<func::FuncOp>();
      llvm::errs() << "cleave-parallelize-loops: " << (func ? func.getName() : StringRef("?")) << ": " << why << '\n';
    }
    return false;
  };
  std::optional<SmallVector<Value>> written = privateWrites(loop, module, fresh, refuse);
  if (!written)
    return false;
  Value iv = loop.getInductionVar();
  for (Value memref : *written) {
    bool ok = true;
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
    if (!ok && !accessesAreDisjoint(loop, {loop.getInductionVar()}, memref))
      return refuse("two iterations may touch an element of a written buffer");
  }
  return true;
}

// The loop perfectly nested in `loop`, if it can join it in a parallel band:
// the only loop of its body, every other operation pure and without regions
// (moved into it when the band is made parallel, recomputed by each of its
// iterations), carrying nothing, and bounds that aren't computed in the
// band, constants or values defined before `band`'s outermost: an
// `scf.parallel` is rectangular.
static scf::ForOp nestedLoop(scf::ForOp loop, scf::ForOp outermost) {
  scf::ForOp nested;
  for (Operation &op : loop.getBody()->without_terminator()) {
    if (auto inner = dyn_cast<scf::ForOp>(op)) {
      if (nested)
        return {};
      nested = inner;
    } else if (op.getNumRegions() != 0 || !isMemoryEffectFree(&op)) {
      return {};
    }
  }
  if (!nested || nested.getNumRegionIterArgs() != 0)
    return {};
  for (Value bound : {nested.getLowerBound(), nested.getUpperBound(), nested.getStep()}) {
    if (getConstantIntValue(bound))
      continue;
    Operation *at = bound.getDefiningOp() ? bound.getDefiningOp() : bound.getParentBlock()->getParentOp();
    if (outermost->isAncestor(at))
      return {};
  }
  return nested;
}

// `loop`, whose iterations are independent, and the loops nested in it
// whose iterations are independent too, in all the band's dimensions at once
// (`accessesAreDisjoint`): the band a kernel runs a thread per innermost
// iteration of. A loop over rows of independent columns (`rope`'s rows,
// heads, pairs) is then one thread per element rather than per row.
static SmallVector<scf::ForOp> parallelBand(scf::ForOp loop, ModuleOp module, FreshMemo &fresh) {
  SmallVector<scf::ForOp> band{loop};
  std::optional<SmallVector<Value>> written = privateWrites(loop, module, fresh, [](const char *) { return false; });
  if (!written)
    return band;
  while (scf::ForOp nested = nestedLoop(band.back(), loop)) {
    band.push_back(nested);
    SmallVector<Value> ivs = llvm::map_to_vector(band, [](scf::ForOp level) { return level.getInductionVar(); });
    if (!llvm::all_of(*written, [&](Value memref) { return accessesAreDisjoint(loop, ivs, memref); })) {
      band.pop_back();
      break;
    }
  }
  return band;
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

// A loop summing a nested loop's terms into what it carries (`for i { for j
// { total = total + f(i, j) } }`, `sum` of a matrix) regrouped by rows: each
// row's terms summed from 0 by the inner loop, then added to the total
// (`total = total + (f(i, 0) + f(i, 1) + ...)`), which `splitCarriedChain`
// then splits into the rows' sums, a kernel (a workgroup each,
// `cleave-cooperative-rows`), and the host adding them up: a float sum
// reassociated, as the device's row reductions already are. nanoLM's
// gradient clipping read its squared gradients back, 433 MiB a step, to sum
// them on the host. A copy of `loop` before it, regrouped; null when it isn't
// that shape: one carried float, used only as the initial value of the
// body's loop, which carries it alone and adds to it (`arith.addf`, its only
// use, yielded) a term; the outer loop yields what the inner one gives.
static scf::ForOp regroupNestedSum(scf::ForOp loop) {
  if (loop.getNumRegionIterArgs() != 1 || !isa<FloatType>(loop.getRegionIterArgs()[0].getType()))
    return {};
  Value total = loop.getRegionIterArgs()[0];
  if (!total.hasOneUse())
    return {};
  auto inner = dyn_cast<scf::ForOp>(total.use_begin()->getOwner());
  if (!inner || inner->getBlock() != loop.getBody() || inner.getNumRegionIterArgs() != 1 ||
      inner.getInitArgs()[0] != total || loop.getBody()->getTerminator()->getOperand(0) != inner.getResult(0))
    return {};
  Value partial = inner.getRegionIterArgs()[0];
  if (!partial.hasOneUse())
    return {};
  auto add = dyn_cast<arith::AddFOp>(partial.use_begin()->getOwner());
  if (!add || add->getBlock() != inner.getBody() || !add->hasOneUse() ||
      inner.getBody()->getTerminator()->getOperand(0) != add.getResult())
    return {};
  OpBuilder builder(loop);
  IRMapping mapping;
  auto copy = cast<scf::ForOp>(builder.clone(*loop, mapping));
  auto innerCopy = cast<scf::ForOp>(mapping.lookup(inner.getOperation()));
  Location loc = loop.getLoc();
  OpBuilder at(innerCopy);
  Type type = total.getType();
  innerCopy->setOperand(innerCopy.getNumControlOperands(),
                        arith::ConstantOp::create(at, loc, type, FloatAttr::get(type, 0.0)));
  at.setInsertionPointAfter(innerCopy);
  Value sum =
      arith::AddFOp::create(at, loc, copy.getRegionIterArgs()[0], innerCopy.getResult(0), add.getFastmathAttr());
  copy.getBody()->getTerminator()->setOperand(0, sum);
  return copy;
}

// A loop over rows `r` accumulating into elements a loop over `c` in it
// picks by `c` at one index position, the others anything the row computes
// (`for r { s = f(r); for c { d[c] = d[c] + g(r, c, s) } }`, `rms_norm_grad_g`,
// a bias's gradient; `g[ids[r], c] += u[r, c]`, an embedding's, rows of the
// table hit by several rows), its order of loops exchanged:
// `perRow` computes each row's values the inner loop needs (`s`, stored in
// `buffers` by row; an index computed from `r` alone is recomputed instead),
// `perColumn` then runs, for each `c`, the rows in their order. The same
// updates of each element in the same order: the same result to the bit,
// while both loops may have independent iterations where the original's
// rows all wrote the same elements. `loop` counts from 0 by 1 to a constant;
// the rows' part reads memory and computes, writes nothing; the inner loop
// carries nothing and its bounds don't depend on the row. `erase` undoes it.
struct Interchange {
  scf::ForOp perRow, perColumn;
  SmallVector<Operation *> buffers, frees;
  void erase() {
    for (Operation *op : frees)
      op->erase();
    perColumn.erase();
    perRow.erase();
    for (Operation *op : buffers)
      op->erase();
  }
};

static std::optional<Interchange> interchangeAccumulation(scf::ForOp loop) {
  if (loop.getNumRegionIterArgs() != 0)
    return std::nullopt;
  std::optional<int64_t> lower = getConstantIntValue(loop.getLowerBound()),
                         upper = getConstantIntValue(loop.getUpperBound()),
                         step = getConstantIntValue(loop.getStep());
  if (!lower || *lower != 0 || !step || *step != 1 || !upper || *upper <= 0)
    return std::nullopt;
  Block *body = loop.getBody();
  auto ops = body->without_terminator();
  if (ops.empty())
    return std::nullopt;
  auto inner = dyn_cast<scf::ForOp>(std::prev(ops.end()));
  if (!inner || inner.getNumRegionIterArgs() != 0)
    return std::nullopt;
  for (Value bound : {inner.getLowerBound(), inner.getUpperBound(), inner.getStep()})
    if (Operation *def = bound.getDefiningOp(); def && loop->isAncestor(def))
      return std::nullopt;
  // The rows' part writes nothing.
  for (Operation &op : ops) {
    if (&op == inner.getOperation())
      continue;
    bool writes = false;
    op.walk([&](Operation *nested) {
      if (!isMemoryEffectFree(nested) && !isa<memref::LoadOp>(nested) &&
          !nested->hasTrait<OpTrait::HasRecursiveMemoryEffects>())
        writes = true;
    });
    if (writes)
      return std::nullopt;
  }
  // What depends on the row.
  llvm::DenseSet<Value> onRow{loop.getInductionVar()};
  body->walk([&](Operation *op) {
    if (llvm::any_of(op->getOperands(), [&](Value v) { return onRow.contains(v); }))
      for (Value result : op->getResults())
        onRow.insert(result);
  });
  // What the inner loop writes: reached with its own variable at one
  // position, which no row's value gives (each `c` its own elements, every
  // update of one by the rows in their order); nothing in the rows' part
  // reads it.
  Value c = inner.getInductionVar();
  llvm::DenseSet<Value> written;
  bool ok = true;
  inner.getBody()->walk([&](Operation *op) {
    if (isa<CallOpInterface>(op))
      ok = false;
    else if (auto store = dyn_cast<memref::StoreOp>(op))
      written.insert(viewRoot(store.getMemRef()));
    else if (!isa<memref::LoadOp>(op) && !isMemoryEffectFree(op) &&
             !op->hasTrait<OpTrait::HasRecursiveMemoryEffects>())
      ok = false;
  });
  if (!ok || written.empty())
    return std::nullopt;
  llvm::DenseMap<Value, unsigned> position;
  body->walk([&](Operation *op) {
    Value accessed;
    ValueRange indices;
    if (auto load = dyn_cast<memref::LoadOp>(op))
      accessed = load.getMemRef(), indices = load.getIndices();
    else if (auto store = dyn_cast<memref::StoreOp>(op))
      accessed = store.getMemRef(), indices = store.getIndices();
    if (!accessed || !written.contains(viewRoot(accessed)))
      return;
    if (!inner->isAncestor(op) || accessed != viewRoot(accessed)) {
      ok = false;
      return;
    }
    std::optional<unsigned> here;
    for (auto [i, index] : llvm::enumerate(indices))
      if (isInductionVariable(index, c))
        here = i;
    auto [it, fresh] = position.try_emplace(accessed, here.value_or(~0u));
    if (!here || (!fresh && it->second != *here))
      ok = false;
  });
  if (!ok)
    return std::nullopt;
  // The rows' values the inner loop uses: what operations without memory
  // effects compute recomputed, from the row, constants and the results of
  // the others (a load, a reduction's loop: `ids[r]` under its index cast),
  // which are stored by row (32-bit integers and floats only).
  SetVector<Value> crossing;
  inner.getBody()->walk([&](Operation *op) {
    for (Value operand : op->getOperands())
      if (Operation *def = operand.getDefiningOp(); def && def->getBlock() == body)
        crossing.insert(operand);
  });
  llvm::DenseSet<Operation *> recompute;
  SetVector<Value> stored;
  SmallVector<Value> pending(crossing.begin(), crossing.end());
  llvm::DenseSet<Value> seen;
  while (!pending.empty()) {
    Value v = pending.pop_back_val();
    if (!seen.insert(v).second)
      continue;
    Operation *def = v.getDefiningOp();
    if (!def || def->getBlock() != body)
      continue;
    if (isPure(def) && def->getNumRegions() == 0) {
      recompute.insert(def);
      for (Value operand : def->getOperands())
        pending.push_back(operand);
      continue;
    }
    if (!v.getType().isIntOrFloat() || v.getType().getIntOrFloatBitWidth() != 32)
      return std::nullopt;
    stored.insert(v);
  }

  OpBuilder builder(loop);
  Location loc = loop.getLoc();
  Interchange result;
  SmallVector<Value> buffers;
  for (Value v : stored) {
    auto alloc = memref::AllocOp::create(builder, loc, MemRefType::get({*upper}, v.getType()));
    result.buffers.push_back(alloc);
    buffers.push_back(alloc);
  }
  auto asIndex = [&](OpBuilder &b, Value iv) -> Value {
    return iv.getType().isIndex() ? iv : arith::IndexCastOp::create(b, loc, b.getIndexType(), iv).getResult();
  };
  result.perRow = scf::ForOp::create(
      builder, loc, loop.getLowerBound(), loop.getUpperBound(), loop.getStep(), ValueRange{},
      [&](OpBuilder &b, Location, Value r, ValueRange) {
        IRMapping mapping;
        mapping.map(loop.getInductionVar(), r);
        for (Operation &op : ops)
          if (&op != inner.getOperation())
            b.clone(op, mapping);
        Value at = asIndex(b, r);
        for (auto [v, buffer] : llvm::zip(stored, buffers))
          memref::StoreOp::create(b, loc, mapping.lookup(v), buffer, ValueRange{at});
        scf::YieldOp::create(b, loc);
      });
  result.perColumn = scf::ForOp::create(
      builder, loc, inner.getLowerBound(), inner.getUpperBound(), inner.getStep(), ValueRange{},
      [&](OpBuilder &b, Location, Value column, ValueRange) {
        scf::ForOp::create(b, loc, loop.getLowerBound(), loop.getUpperBound(), loop.getStep(), ValueRange{},
                           [&](OpBuilder &rb, Location, Value r, ValueRange) {
                             IRMapping mapping;
                             mapping.map(loop.getInductionVar(), r);
                             mapping.map(c, column);
                             Value at = asIndex(rb, r);
                             for (auto [v, buffer] : llvm::zip(stored, buffers))
                               mapping.map(v, memref::LoadOp::create(rb, loc, buffer, ValueRange{at}).getResult());
                             for (Operation &op : ops)
                               if (recompute.contains(&op))
                                 rb.clone(op, mapping);
                             for (Operation &op : inner.getBody()->without_terminator())
                               rb.clone(op, mapping);
                             scf::YieldOp::create(rb, loc);
                           });
        scf::YieldOp::create(b, loc);
      });
  for (Value buffer : buffers)
    result.frees.push_back(memref::DeallocOp::create(builder, loc, buffer));
  return result;
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
// the work parallelized if its iterations are independent, a nested sum
// first regrouped by rows (`regroupNestedSum`); one accumulating
// over its rows into what an inner loop picks by column, its loops exchanged
// (`interchangeAccumulation`), kept so if both halves are independent.
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
    if (scf::ForOp regrouped = regroupNestedSum(copy ? copy : loop)) {
      std::optional<CarriedSplit> split = splitCarriedChain(regrouped);
      if (split && iterationsAreIndependent(split->parallel, module, fresh)) {
        loop->replaceAllUsesWith(split->sequential.getResults());
        loop.erase();
        if (copy)
          copy.erase();
        regrouped.erase();
        fresh.clear();
        candidates.push_back(split->parallel);
        continue;
      }
      if (split)
        split->erase();
      regrouped.erase();
      fresh.clear();
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
    if (std::optional<Interchange> exchanged = interchangeAccumulation(copy ? copy : loop)) {
      if (iterationsAreIndependent(exchanged->perRow, module, fresh) &&
          iterationsAreIndependent(exchanged->perColumn, module, fresh)) {
        loop.erase();
        if (copy)
          copy.erase();
        fresh.clear();
        candidates.push_back(exchanged->perRow);
        candidates.push_back(exchanged->perColumn);
        continue;
      }
      exchanged->erase();
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
  // Every band found before any is rewritten: `fresh` remembers values.
  SmallVector<SmallVector<scf::ForOp>> bands;
  for (scf::ForOp loop : candidates)
    bands.push_back(parallelBand(loop, module, fresh));
  for (MutableArrayRef<scf::ForOp> band : bands) {
    scf::ForOp loop = band.front();
    if (debuggingParallelization())
      llvm::errs() << "cleave-parallelize-loops: parallel in " << loop->getParentOfType<func::FuncOp>().getName()
                   << ", " << band.size() << " loops deep, at " << loop.getLoc() << '\n';
    // The band made a perfect nest: what each loop's body computes before or
    // after the next loop (pure, `nestedLoop`) moved to the start of the
    // next's body, in order.
    for (auto [outer, inner] : llvm::zip(band.drop_back(), band.drop_front())) {
      Operation *first = &inner.getBody()->front();
      for (Operation &op : llvm::make_early_inc_range(outer.getBody()->without_terminator()))
        if (&op != inner.getOperation())
          op.moveBefore(first);
    }
    rewriter.setInsertionPoint(loop);
    Location loc = loop.getLoc();
    Type index = rewriter.getIndexType();
    auto asIndex = [&](Value v) -> Value {
      if (std::optional<int64_t> c = getConstantIntValue(v))
        return arith::ConstantIndexOp::create(rewriter, loc, *c);
      return v.getType().isIndex() ? v : arith::IndexCastOp::create(rewriter, loc, index, v).getResult();
    };
    SmallVector<Value> lower, upper, steps;
    for (scf::ForOp level : band) {
      lower.push_back(asIndex(level.getLowerBound()));
      upper.push_back(asIndex(level.getUpperBound()));
      steps.push_back(asIndex(level.getStep()));
    }
    auto parallel = scf::ParallelOp::create(rewriter, loc, lower, upper, steps);
    Block *body = parallel.getBody();
    rewriter.setInsertionPointToStart(body);
    SmallVector<Value> originals;
    for (auto [level, iv] : llvm::zip(band, body->getArguments())) {
      Type type = level.getInductionVar().getType();
      originals.push_back(type.isIndex() ? iv : arith::IndexCastOp::create(rewriter, loc, type, iv).getResult());
    }
    for (auto [level, original] : llvm::zip(band.drop_back(), originals))
      level.getInductionVar().replaceAllUsesWith(original);
    // The innermost loop's body, its `scf.yield` dropped, before the parallel
    // loop's terminator; the band's loops, emptied, erased.
    Block *innermost = band.back().getBody();
    rewriter.eraseOp(innermost->getTerminator());
    rewriter.inlineBlockBefore(innermost, body->getTerminator(), ValueRange{originals.back()});
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

// Whether the memory of `memref` (a local allocation, or a parameter it is
// passed as) never reaches the device: every use an access, a view, its
// metadata or deallocation, an address only compared (`addressOnlyCompared`),
// or a call to a function of the module that launches nothing (`launching`
// holds those that may), its parameter used the same way. The host's
// accesses to it then need no report: bufferization's deallocation fills
// small arrays of addresses and flags every iteration of a loop
// (`dealloc_helper`), 400k reports a nanoLM step.
static bool staysOnHost(Value memref, ModuleOp module, const llvm::DenseSet<Operation *> &launching, int depth) {
  if (depth == 0)
    return false;
  Value root = viewRoot(memref);
  if (!root.getDefiningOp<memref::AllocOp>() && !isa<BlockArgument>(root))
    return false;
  if (auto arg = dyn_cast<BlockArgument>(root)) {
    auto func = dyn_cast<func::FuncOp>(arg.getOwner()->getParentOp());
    if (!func || func.isPublic() || launching.contains(func))
      return false;
  }
  SmallVector<Value> pending{root};
  while (!pending.empty()) {
    Value v = pending.pop_back_val();
    for (OpOperand &use : v.getUses()) {
      Operation *user = use.getOwner();
      if (isa<memref::LoadOp, memref::StoreOp, memref::DeallocOp, memref::DimOp>(user))
        continue;
      if (isa<ViewLikeOpInterface, memref::CastOp>(user)) {
        pending.push_back(user->getResult(0));
        continue;
      }
      if (auto metadata = dyn_cast<memref::ExtractStridedMetadataOp>(user)) {
        pending.push_back(metadata.getBaseBuffer());
        continue;
      }
      if (auto extract = dyn_cast<memref::ExtractAlignedPointerAsIndexOp>(user)) {
        if (extract->hasAttr(kRuntimeAddress) || !addressOnlyCompared(extract.getResult(), module, 6))
          return false;
        continue;
      }
      if (auto call = dyn_cast<func::CallOp>(user)) {
        auto callee = module.lookupSymbol<func::FuncOp>(call.getCallee());
        if (callee && !callee.isExternal() && !launching.contains(callee) &&
            staysOnHost(callee.getArgument(use.getOperandNumber()), module, launching, depth - 1))
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
  // The runtime's calls that launch device work: a kernel, or a bf16 product
  // (`cleave-lower-gpu-gemms`).
  auto isLaunch = [&](StringRef callee) {
    return callee == dispatchName || callee == kGemmBf16 || callee == kGemmBf16Operands;
  };
  if (!module.lookupSymbol<func::FuncOp>(dispatchName) && !module.lookupSymbol<func::FuncOp>(kGemmBf16) &&
      !module.lookupSymbol<func::FuncOp>(kGemmBf16Operands))
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
        if (isLaunch(call.getCallee()) || (callee && launching.contains(callee)))
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
        if (isLaunch(call.getCallee()) || (callee && launching.contains(callee)))
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
  llvm::DenseMap<Value, bool> hostOnly;
  auto record = [&](Operation *at, Value v, int access) {
    if (!isa<MemRefType>(v.getType()))
      return; // an unranked memref: not lowered by this backend
    Value root = viewRoot(v);
    // The stack or a global: no mirror covers it (the runtime mirrors heap
    // payloads only), whatever a kernel does with it, a copy each time (the
    // runtime's own argument arrays, filled before every dispatch).
    if (root.getDefiningOp<memref::AllocaOp>() || root.getDefiningOp<memref::GetGlobalOp>())
      return;
    auto [known, inserted] = hostOnly.try_emplace(root, false);
    if (inserted)
      known->second = staysOnHost(root, module, launching, 4);
    if (known->second)
      return;
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
// Whether `cast` makes a memref of a descriptor built around a fresh heap
// allocation: its allocated and aligned pointers one `cleave_alloc_rc`
// result, used for nothing else, at offset 0; the descriptor converted to a
// memref by this cast alone, put anywhere else (a struct's field: a
// gradient's) only after it. What `mlir_lower.rs` makes for a tensor field
// a function's result is written into: as fresh as a `memref.alloc`.
static bool isFreshDescriptor(UnrealizedConversionCastOp cast, SmallVectorImpl<Value> *aliases = nullptr,
                              SmallVectorImpl<Operation *> *escapes = nullptr, bool arena = false) {
  if (cast->getNumOperands() != 1 || cast->getNumResults() != 1)
    return false;
  Value descriptor = cast->getOperand(0);
  // Each field's last insertion, from the end of the chain.
  llvm::DenseMap<int64_t, Value> fields;
  SmallVector<Operation *> chain;
  Value at = descriptor;
  while (auto insert = at.getDefiningOp<LLVM::InsertValueOp>()) {
    chain.push_back(insert);
    ArrayRef<int64_t> position = insert.getPosition();
    if (position.size() == 1)
      fields.try_emplace(position[0], insert.getValue());
    at = insert.getContainer();
  }
  Value allocated = fields.lookup(0), aligned = fields.lookup(1), offset = fields.lookup(2);
  if (!allocated || allocated != aligned || !offset || getConstantIntValue(offset) != std::optional<int64_t>(0))
    return false;
  // `arena`: a block of the thread's arena too (`cleave_alloc_local`, a
  // result that doesn't leave its caller's loop), as fresh.
  auto allocator = [&](StringRef callee) {
    return callee == "cleave_alloc_rc" || (arena && callee == "cleave_alloc_local");
  };
  bool fromAllocator = false;
  if (auto call = allocated.getDefiningOp<func::CallOp>())
    fromAllocator = allocator(call.getCallee());
  else if (auto call = allocated.getDefiningOp<LLVM::CallOp>())
    fromAllocator = call.getCallee() && allocator(*call.getCallee());
  if (!fromAllocator)
    return false;
  // Its other uses only count references (the header before the payload,
  // never the payload: a field read twice is retained).
  auto refcounting = [](Operation *user) {
    auto call = dyn_cast<func::CallOp>(user);
    return call && (call.getCallee() == "cleave_retain" || call.getCallee() == "cleave_release" ||
                    call.getCallee() == "cleave_release_void");
  };
  for (Operation *user : allocated.getUsers())
    if (!llvm::is_contained(chain, user) && !refcounting(user))
      return false;
  // Other casts of the descriptor (`mlir_lower.rs` converts it again where it
  // reads the field) are other names of the memory, and anything else using
  // it (put in a struct, which a call may then read or write) reaches it
  // too, all after this cast: other casts allowed only when the caller counts
  // their uses (`aliases`); the rest reported in `escapes` when asked (a
  // caller that only needs nothing to come before the cast doesn't).
  for (Operation *user : descriptor.getUsers()) {
    if (user == cast.getOperation())
      continue;
    if (user->getBlock() != cast->getBlock() || user->isBeforeInBlock(cast))
      return false;
    if (isa<UnrealizedConversionCastOp>(user)) {
      if (!aliases)
        return false;
      aliases->push_back(user->getResult(0));
    } else if (escapes) {
      escapes->push_back(user);
    }
  }
  return true;
}

static bool isFreshUnaliased(Value v, ModuleOp module, FreshMemo &memo) {
  if (v.getDefiningOp<memref::AllocOp>())
    return true;
  if (auto cast = v.getDefiningOp<UnrealizedConversionCastOp>())
    return isFreshDescriptor(cast);
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
// Whether `v`, a value in `loop`, is computed from values defined outside it
// by pure operations without regions (an index cast of the outer loop's
// variable, recomputed in the inner loop's body), so the same before it.
static bool isInvariantIn(Value v, scf::ForOp loop) {
  Operation *def = v.getDefiningOp();
  if (!def)
    return !loop->isAncestor(v.getParentBlock()->getParentOp());
  if (!loop->isAncestor(def))
    return true;
  return def->getNumRegions() == 0 && isPure(def) &&
         llvm::all_of(def->getOperands(), [&](Value operand) { return isInvariantIn(operand, loop); });
}

// Whether `a` and `b` are the same value: one, or the results of operations
// alike computing the same values (two index casts of one variable).
static bool sameValue(Value a, Value b) {
  if (a == b)
    return true;
  auto ra = dyn_cast<OpResult>(a), rb = dyn_cast<OpResult>(b);
  if (!ra || !rb || ra.getResultNumber() != rb.getResultNumber())
    return false;
  Operation *x = ra.getOwner(), *y = rb.getOwner();
  return x->getName() == y->getName() && x->getAttrDictionary() == y->getAttrDictionary() &&
         x->getNumRegions() == 0 && y->getNumRegions() == 0 && isPure(x) && isPure(y) &&
         x->getNumOperands() == y->getNumOperands() && x->getResultTypes() == y->getResultTypes() &&
         llvm::all_of(llvm::zip(x->getOperands(), y->getOperands()),
                      [](auto pair) { return sameValue(std::get<0>(pair), std::get<1>(pair)); });
}

// `v` (`isInvariantIn(v, loop)`) at `builder`'s insertion point, before the
// loop: itself, or its computation cloned there.
static Value cloneBefore(Value v, scf::ForOp loop, OpBuilder &builder, IRMapping &mapping) {
  Operation *def = v.getDefiningOp();
  if (!def || !loop->isAncestor(def))
    return v;
  if (Value known = mapping.lookupOrNull(v))
    return known;
  for (Value operand : def->getOperands())
    mapping.map(operand, cloneBefore(operand, loop, builder, mapping));
  Operation *clone = builder.clone(*def, mapping);
  return clone->getResult(cast<OpResult>(v).getResultNumber());
}

// Whether `root` is fresh memory as far as `loop` sees: `isFreshUnaliased`,
// or a fresh descriptor also cast to other names (`isFreshDescriptor`'s
// aliases: a gradient's field, released or read after the loop, `reduce0`'s
// bias gradient) that the loop itself never reaches.
static bool freshInLoop(Value root, scf::ForOp loop, ModuleOp module, FreshMemo &fresh) {
  if (isFreshUnaliased(root, module, fresh))
    return true;
  auto cast = root.getDefiningOp<UnrealizedConversionCastOp>();
  SmallVector<Value> aliases;
  if (!cast || !isFreshDescriptor(cast, &aliases))
    return false;
  bool reached = false;
  loop.getBody()->walk([&](Operation *inner) {
    for (Value operand : inner->getOperands())
      reached |= isa<BaseMemRefType>(operand.getType()) && llvm::is_contained(aliases, viewRoot(operand));
  });
  return !reached;
}

// A loop accumulating into one element of memory (`for r { d[c] = d[c] +
// f(r) }`, `interchangeAccumulation`'s columns) made to carry it instead:
// loaded before the loop, the loop's value, stored after. `cleaveCooperativeRows`
// then shares the loop out, a workgroup per column rather than a thread (12
// workgroups for `rms_norm_grad_g`'s 768 columns, the costliest of nanoLM's
// kernels). The loop carries nothing; its body's one store, at indices
// the same in every iteration (`isInvariantIn`), follows a load of the same
// element (`sameValue`) in the body
// itself; the memory is a fresh allocation's (`freshInLoop`: nothing
// else the loop reads aliases it) that nothing else in the loop reaches, and
// the loop writes nothing else. Whether it changed `loop`.
static bool promoteAccumulation(scf::ForOp loop, ModuleOp module, FreshMemo &fresh) {
  if (loop.getNumRegionIterArgs() != 0)
    return false;
  memref::StoreOp store;
  bool ok = true;
  loop.getBody()->walk([&](Operation *inner) {
    if (auto candidate = dyn_cast<memref::StoreOp>(inner)) {
      ok &= !store && candidate->getBlock() == loop.getBody();
      store = candidate;
    } else if (!isa<memref::LoadOp>(inner) && !isMemoryEffectFree(inner) &&
               !inner->hasTrait<OpTrait::HasRecursiveMemoryEffects>()) {
      ok = false;
    }
  });
  if (!ok || !store)
    return false;
  Value root = viewRoot(store.getMemRef());
  if (!freshInLoop(root, loop, module, fresh) ||
      !llvm::all_of(store.getIndices(), [&](Value index) { return isInvariantIn(index, loop); }))
    return false;
  memref::LoadOp load;
  loop.getBody()->walk([&](memref::LoadOp candidate) {
    if (viewRoot(candidate.getMemRef()) != root)
      return;
    bool same = candidate.getMemRef() == store.getMemRef() && candidate->getBlock() == loop.getBody() &&
                candidate->isBeforeInBlock(store) && candidate.getIndices().size() == store.getIndices().size() &&
                llvm::all_of(llvm::zip(candidate.getIndices(), store.getIndices()),
                             [](auto pair) { return sameValue(std::get<0>(pair), std::get<1>(pair)); });
    ok &= same && !load;
    load = candidate;
  });
  if (!ok || !load)
    return false;
  IRRewriter rewriter(loop.getContext());
  rewriter.setInsertionPoint(loop);
  Location loc = loop.getLoc();
  IRMapping hoisted;
  SmallVector<Value> indices;
  for (Value index : store.getIndices())
    indices.push_back(cloneBefore(index, loop, rewriter, hoisted));
  Value initial = memref::LoadOp::create(rewriter, loc, store.getMemRef(), indices);
  Value stored = store.getValueToStore();
  FailureOr<LoopLikeOpInterface> replaced = cast<LoopLikeOpInterface>(loop.getOperation())
                                                .replaceWithAdditionalYields(
                                                    rewriter, ValueRange{initial}, /*replaceInitOperandUsesInLoop=*/false,
                                                    [&](OpBuilder &, Location, ArrayRef<BlockArgument>) {
                                                      return SmallVector<Value>{stored};
                                                    });
  if (failed(replaced))
    return false;
  auto carrying = cast<scf::ForOp>(replaced->getOperation());
  load.replaceAllUsesWith(carrying.getRegionIterArgs().back());
  rewriter.setInsertionPointAfter(carrying);
  memref::StoreOp::create(rewriter, loc, carrying.getResults().back(), store.getMemRef(), indices);
  rewriter.eraseOp(store);
  rewriter.eraseOp(load);
  return true;
}

static int64_t cleaveCooperativeRows(MlirOperation op, int64_t threads) {
  auto module = dyn_cast<ModuleOp>(unwrap(op));
  if (!module)
    return 0;
  MLIRContext *ctx = module.getContext();
  // Row loops' accumulations into memory made carried values first.
  FreshMemo fresh;
  module.walk([&](scf::ParallelOp loop) {
    if (loop.getNumLoops() != 1 || loop.getNumResults() != 0 || loop->getParentOfType<gpu::GPUModuleOp>() ||
        loop->getParentOfType<gpu::LaunchOp>() || loop->getParentOfType<scf::ParallelOp>())
      return;
    SmallVector<scf::ForOp> inner;
    for (Operation &op : loop.getBody()->without_terminator())
      if (auto forOp = dyn_cast<scf::ForOp>(op))
        inner.push_back(forOp);
    for (scf::ForOp forOp : inner)
      if (promoteAccumulation(forOp, module, fresh))
        fresh.clear();
  });
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
  // Which operands are read from device memory 4 elements at a time
  // (`vector<4xf32>` loads), where they can be: bit 1 those whose rows are
  // contiguous, bit 2 those whose `k` is.
  int64_t vec = 0;
  // Whether workgroup memory holds two slices, the next one stored while the
  // current one is computed with: one barrier a slice instead of two.
  int64_t db = 0;
  int64_t threads() const { return (bm / tm) * (bn / tn); }
  // Whether it can compute an `m` x `n` product over `k`: tiles dividing it,
  // each thread a whole block and a whole share of each slice's loads, a
  // workgroup of at most 1024 threads, its workgroup memory (two `bk` x `bm`
  // and `bk` x `bn` slices, padded) within 48 KiB.
  bool fits(int64_t m, int64_t n, int64_t k) const {
    if (bm <= 0 || bn <= 0 || bk <= 0 || tm <= 0 || tn <= 0 || bm % tm != 0 || bn % tn != 0)
      return false;
    int64_t t = threads();
    int64_t shared = (db ? 2 : 1) * 4 * bk * ((bm + 4) + (bn + 4));
    return m % bm == 0 && n % bn == 0 && k % bk == 0 && t >= 32 && t <= 1024 && (bm * bk) % t == 0 &&
           (bn * bk) % t == 0 && shared <= 48 * 1024;
  }
};

// `CLEAVE_GPU_MATMUL=bm,bn,bk,tm,tn,split[,vec[,db]]` (read at compile time):
// the tiling and split of every product it fits (`ProductTiling::fits`),
// split 0 chosen as without it: what a tuner tries one configuration after
// another (`bench/gpu_matmul/tune.rs`).
struct ProductConfig {
  ProductTiling tiling;
  int64_t split;
};
static std::optional<ProductConfig> forcedProductConfig() {
  static const std::optional<ProductConfig> config = []() -> std::optional<ProductConfig> {
    const char *v = std::getenv("CLEAVE_GPU_MATMUL");
    if (!v)
      return std::nullopt;
    SmallVector<StringRef> fields;
    StringRef(v).split(fields, ',');
    SmallVector<int64_t> values;
    for (StringRef field : fields) {
      int64_t value;
      if (field.trim().getAsInteger(10, value))
        return std::nullopt;
      values.push_back(value);
    }
    if (values.size() < 6 || values.size() > 8)
      return std::nullopt;
    values.resize(8, 0);
    return ProductConfig{{values[0], values[1], values[2], values[3], values[4], values[6], values[7]}, values[5]};
  }();
  return config;
}

// The tuned products' table (`matmul_table.inc`, written by
// `bench/gpu_matmul/tune.rs`, or the file `CLEAVE_GPU_MATMUL_TABLE` names at
// compile time): the fastest configuration measured for each shape and
// layout (whether `k` is contiguous in A and in B), `vec` and `db` last (0
// when left out).
static const char *const kTunedProducts =
#include "matmul_table.inc"
    ;
struct TunedProduct {
  int64_t m, n, k;
  bool aK, bK;
  ProductConfig config;
};
static const std::vector<TunedProduct> &tunedProducts() {
  static const std::vector<TunedProduct> table = [] {
    std::string text = kTunedProducts;
    if (const char *path = std::getenv("CLEAVE_GPU_MATMUL_TABLE")) {
      if (auto file = llvm::MemoryBuffer::getFile(path))
        text = (*file)->getBuffer().str();
    }
    std::vector<TunedProduct> entries;
    SmallVector<StringRef> lines;
    StringRef(text).split(lines, '\n');
    for (StringRef line : lines) {
      SmallVector<StringRef> fields;
      line.trim().split(fields, ' ', -1, /*KeepEmpty=*/false);
      if (fields.size() < 11 || fields.size() > 13)
        continue;
      int64_t v[13] = {};
      bool bad = false;
      for (size_t i = 0; i < fields.size(); ++i)
        bad |= fields[i].getAsInteger(10, v[i]);
      if (!bad)
        entries.push_back(
            {v[0], v[1], v[2], v[3] != 0, v[4] != 0, {{v[5], v[6], v[7], v[8], v[9], v[11], v[12]}, v[10]}});
    }
    return entries;
  }();
  return table;
}

// The tuned configuration of an `m` x `n` product over `k` of that layout:
// its own shape's, else the nearest tuned shape's (each dimension within a
// factor of 2) whose configuration fits it; none otherwise.
static std::optional<ProductConfig> tunedProductConfig(int64_t m, int64_t n, int64_t k, bool aK, bool bK) {
  std::optional<ProductConfig> chosen;
  double nearest = 1e30;
  for (const TunedProduct &entry : tunedProducts()) {
    if (entry.aK != aK || entry.bK != bK || !entry.config.tiling.fits(m, n, k) ||
        (entry.config.split > 1 && k % (entry.config.split * entry.config.tiling.bk) != 0))
      continue;
    double distance = 0;
    bool near = true;
    for (auto [mine, theirs] : {std::pair{m, entry.m}, std::pair{n, entry.n}, std::pair{k, entry.k}}) {
      double ratio = std::log2(double(mine) / double(theirs));
      near &= std::abs(ratio) <= 1.0;
      distance += std::abs(ratio);
    }
    if (near && distance < nearest) {
      nearest = distance;
      chosen = entry.config;
    }
  }
  return chosen;
}

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
// What reads and writes `memref`, a `memref.alloc` of the function, if that
// is all that uses it (besides `product`, its deallocation and its address
// taken for the runtime): direct loads and stores, fills of the whole of
// it, nothing that could reach it otherwise (a view, a call, another
// `linalg` op). None otherwise.
struct DirectUses {
  SmallVector<memref::LoadOp> loads;
  SmallVector<memref::StoreOp> stores;
  SmallVector<linalg::FillOp> fills;
  // `linalg` ops reading it as an input only.
  SmallVector<Operation *> readers;
};
static std::optional<DirectUses> directUses(Value memref, Operation *product) {
  // Through `memref.cast`s (a layout made dynamic for `sgemm`), which index
  // the same elements the same way: the allocation and every cast of it.
  while (auto cast = memref.getDefiningOp<memref::CastOp>())
    memref = cast.getSource();
  if (!memref.getDefiningOp<memref::AllocOp>())
    return std::nullopt;
  DirectUses uses;
  SmallVector<Value> names{memref};
  for (size_t i = 0; i < names.size(); ++i)
    for (Operation *user : names[i].getUsers())
      if (auto cast = dyn_cast<memref::CastOp>(user))
        names.push_back(cast.getResult());
  for (Value name : names)
  for (OpOperand &use : name.getUses()) {
    Operation *user = use.getOwner();
    if (isa<memref::CastOp>(user))
      continue;
    if (user == product ||
        isa<memref::DeallocOp, memref::ExtractAlignedPointerAsIndexOp, memref::ExtractStridedMetadataOp>(user))
      continue;
    if (auto load = dyn_cast<memref::LoadOp>(user))
      uses.loads.push_back(load);
    else if (auto store = dyn_cast<memref::StoreOp>(user); store && store.getMemRef() == name)
      uses.stores.push_back(store);
    else if (auto fill = dyn_cast<linalg::FillOp>(user); fill && fill.hasPureBufferSemantics())
      uses.fills.push_back(fill);
    else if (auto linalgOp = dyn_cast<linalg::LinalgOp>(user);
             linalgOp && linalgOp.hasPureBufferSemantics() && linalgOp.isDpsInput(&use))
      uses.readers.push_back(user);
    else
      return std::nullopt;
  }
  return uses;
}

static bool runsBefore(Operation *op, Operation *later);

// Whether every element of `memref` in `box` (each index's lowest and
// highest) is zero when `reader` reads it, or never written: what writes
// there before it (`directUses`, all of it before `reader`) writes zero, a
// fill or a store of 0, or a product (`arith.mulf`) one of whose factors is
// a load, at the store's own indices, of memory zero there in turn (`depth`
// levels: attention's score gradient `p * (dp - dot) * scale`, zero where the
// weights `p` are), its other factors assumed finite.
static bool zeroOn(Value memref, ArrayRef<std::pair<int64_t, int64_t>> box, Operation *reader, func::FuncOp func,
                   int depth) {
  std::optional<DirectUses> uses = directUses(memref, nullptr);
  if (!uses || !llvm::all_of(uses->fills, [](linalg::FillOp fill) {
        return matchPattern(fill.getInputs()[0], m_AnyZeroFloat());
      }))
    return false;
  for (memref::StoreOp store : uses->stores) {
    if (!runsBefore(store, reader))
      return false;
    Value value = store.getValueToStore();
    if (matchPattern(value, m_AnyZeroFloat()) || !mayReach(func, store.getIndices(), box))
      continue;
    // A factor loaded where it's stored, from memory zero on the box.
    std::function<bool(Value)> zeroFactor = [&](Value v) {
      if (auto load = v.getDefiningOp<memref::LoadOp>())
        return depth > 0 && load.getIndices().size() == store.getIndices().size() &&
               llvm::all_of(llvm::zip(load.getIndices(), store.getIndices()),
                            [](auto pair) { return sameValue(std::get<0>(pair), std::get<1>(pair)); }) &&
               zeroOn(load.getMemRef(), box, load, func, depth - 1);
      if (auto product = v.getDefiningOp<arith::MulFOp>())
        return zeroFactor(product.getLhs()) || zeroFactor(product.getRhs());
      return false;
    };
    if (!zeroFactor(value))
      return false;
  }
  return true;
}

// Whether `op` runs before `later` whenever both do: an ancestor of each in
// one block, `op`'s first.
static bool runsBefore(Operation *op, Operation *later) {
  for (Operation *at = later; at; at = at->getParentOp()) {
    if (!at->getBlock())
      return false;
    if (Operation *mine = at->getBlock()->findAncestorOpInBlock(*op))
      return mine != at && mine->isBeforeInBlock(at);
  }
  return false;
}

// A call to `cleave_vk_discard` for `memref`, at `builder`'s insertion
// point (declared in `module` if it isn't): its contents are undefined, the
// runtime uploads none of them.
static void discardContents(OpBuilder &builder, Location loc, Value memref, ModuleOp module) {
  StringRef name = "cleave_vk_discard";
  if (!module.lookupSymbol<func::FuncOp>(name)) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(module.getBody());
    Type i64 = builder.getI64Type();
    func::FuncOp::create(builder, builder.getUnknownLoc(), name, builder.getFunctionType({i64, i64}, {})).setPrivate();
  }
  auto [first, bytes] = memrefSpan(builder, loc, memref);
  func::CallOp::create(builder, loc, name, TypeRange{}, ValueRange{first, bytes});
}

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
  // From the most work per thread to the least, the first that divides the
  // product (smaller tiles only for shapes the larger don't divide: they
  // were slower where the larger fit, though more workgroups). A product
  // that still makes too few workgroups to fill the device
  // (`kEnoughWorkgroups`, about two per compute unit: the RTX 4070 SUPER has
  // 56) splits its reduction instead (`split`), when the body allows it.
  const ProductTiling tilings[] = {
      {128, 128, 16, 8, 8}, {64, 64, 16, 4, 4}, {64, 32, 16, 4, 2}, {32, 32, 16, 2, 2}};
  constexpr int64_t kEnoughWorkgroups = 128;
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
    SmallVector<const ProductTiling *> fitting;
    for (const ProductTiling &candidate : tilings) {
      int64_t threads = (candidate.bm / candidate.tm) * (candidate.bn / candidate.tn);
      if (m % candidate.bm == 0 && n % candidate.bn == 0 && k % candidate.bk == 0 &&
          (candidate.bm * candidate.bk) % threads == 0 && (candidate.bn * candidate.bk) % threads == 0)
        fitting.push_back(&candidate);
    }
    if (fitting.empty())
      continue;
    const ProductTiling *tiling = fitting.front();
    // Too few workgroups, and a reduction too short to split (attention's
    // `Q K^T`, `k` 64: 64 workgroups of 128 x 128): a smaller tile, the
    // first that makes enough, else the one that makes the most.
    auto workgroupsOf = [&](const ProductTiling *t) { return (m / t->bm) * (n / t->bn); };
    if (workgroupsOf(tiling) < kEnoughWorkgroups && k < 8 * tiling->bk) {
      for (const ProductTiling *candidate : fitting) {
        if (workgroupsOf(candidate) > workgroupsOf(tiling))
          tiling = candidate;
        if (workgroupsOf(tiling) >= kEnoughWorkgroups)
          break;
      }
    }
    // A configuration forced (`CLEAVE_GPU_MATMUL`), else the tuned one
    // (`tunedProductConfig`), else the choice above.
    std::optional<ProductConfig> forced = forcedProductConfig();
    if (forced && !forced->tiling.fits(m, n, k))
      forced.reset();
    if (!forced)
      forced = tunedProductConfig(m, n, k, product.getMatchingIndexingMap(a).getDimPosition(1) == dimK,
                                  product.getMatchingIndexingMap(b).getDimPosition(1) == dimK);
    if (forced)
      tiling = &forced->tiling;
    const int64_t bm = tiling->bm, bn = tiling->bn, bk = tiling->bk, tm = tiling->tm, tn = tiling->tn;
    const int64_t threadsX = bn / tn, threadsY = bm / tm, threads = threadsX * threadsY;
    // Split-K: the reduction cut in `split` slices, each a workgroup's along
    // `z`, into a buffer of partial results (`partials[slice][m][n]`, each
    // from 0), which a second kernel adds to the result in slice order.
    // Only for a body `c + f(a, b)` (`additive`, `f` not reading `c`), the
    // slices then a regrouping of the sum: the same every run, not the
    // single thread's rounding. Attention's `P V` (1024 x 64 x 1024) made
    // 16 workgroups and a quarter of nanoLM's device time.
    Value yieldedValue = body->getTerminator()->getOperand(0);
    Operation *additive = yieldedValue.getDefiningOp();
    Value cArgument = body->getArgument(2);
    bool splittable = additive && isa<arith::AddFOp>(additive) && additive->getBlock() == body &&
                      (additive->getOperand(0) == cArgument || additive->getOperand(1) == cArgument) &&
                      llvm::all_of(cArgument.getUsers(), [&](Operation *user) { return user == additive; }) &&
                      additive->getOperand(0) != additive->getOperand(1);
    int64_t split = 1;
    const int64_t workgroups = (m / bm) * (n / bn);
    if (forced && forced->split > 0) {
      if (splittable && k % (forced->split * bk) == 0)
        split = forced->split;
    } else {
      while (splittable && workgroups * split < kEnoughWorkgroups && k % (2 * split * bk) == 0 &&
             k / (2 * split) >= 4 * bk)
        split *= 2;
    }
    const int64_t slice = k / split;
    // A fill of the result just before the product (nothing between them
    // touching memory: `sgemm`'s `beta` 0): the value its elements start
    // from instead of the result's, the fill gone (2 of attention's 6
    // kernels per head). Its `cleave_vk_discard`, before it, stays
    // (`cleave-mark-full-overwrites` runs first).
    linalg::FillOp fill;
    for (Operation *before = product->getPrevNode(); before; before = before->getPrevNode()) {
      if (auto candidate = dyn_cast<linalg::FillOp>(before)) {
        if (candidate.hasPureBufferSemantics() && candidate.getOutputs()[0] == c)
          fill = candidate;
        break;
      }
      if (!isMemoryEffectFree(before))
        break;
    }
    Value filled = fill ? fill.getInputs()[0] : Value();
    // The work no one needs, found at compile time (`mayReach`, Presburger
    // arithmetic over the accesses' loops): tiles of the result no load
    // reaches (`causal_softmax` reads the scores below the diagonal only:
    // 28 of `Q K^T`'s 64 tiles), whose workgroups do nothing; and in each row
    // of tiles, the slices of the reduction where `a` is zero (no store of
    // anything but 0 reaches them: the weights above the diagonal), which it
    // doesn't run (`P V`'s and `P^T dO`'s upper halves). The latter assumes `b`
    // finite there (`0 * inf` isn't 0): what's skipped adds exact zeros.
    auto func = product->getParentOfType<func::FuncOp>();
    const int64_t tilesM = m / bm, tilesN = n / bn, slices = k / bk;
    auto boxOf = [](AffineMap map, ArrayRef<std::pair<unsigned, std::pair<int64_t, int64_t>>> ranges) {
      SmallVector<std::pair<int64_t, int64_t>> box;
      for (unsigned r = 0; r < map.getNumResults(); ++r)
        for (auto [dim, range] : ranges)
          if (map.getDimPosition(r) == dim)
            box.push_back(range);
      return box;
    };
    SmallVector<bool> liveTiles;
    if (std::optional<DirectUses> uses = directUses(c, product);
        uses && uses->readers.empty() && tilesM * tilesN * int64_t(uses->loads.size()) <= 4096) {
      for (int64_t tm = 0; tm < tilesM; ++tm)
        for (int64_t tn = 0; tn < tilesN; ++tn) {
          SmallVector<std::pair<int64_t, int64_t>> box =
              boxOf(outMap, {{dimM, {tm * bm, tm * bm + bm - 1}}, {dimN, {tn * bn, tn * bn + bn - 1}}});
          liveTiles.push_back(llvm::any_of(uses->loads, [&](memref::LoadOp load) {
            return mayReach(func, load.getIndices(), box);
          }));
        }
      if (llvm::all_of(liveTiles, [](bool live) { return live; }))
        liveTiles.clear();
    }
    SmallVector<std::pair<int64_t, int64_t>> liveK; // by row of tiles: `[first, end)`
    if (std::optional<DirectUses> uses = directUses(a->get(), product);
        uses && !uses->stores.empty() && tilesM * slices <= 4096) {
      AffineMap aMap = product.getMatchingIndexingMap(a);
      for (int64_t tm = 0; tm < tilesM; ++tm) {
        int64_t first = -1, last = -1;
        for (int64_t slice = 0; slice < slices; ++slice) {
          SmallVector<std::pair<int64_t, int64_t>> box =
              boxOf(aMap, {{dimM, {tm * bm, tm * bm + bm - 1}}, {dimK, {slice * bk, slice * bk + bk - 1}}});
          if (!zeroOn(a->get(), box, product, func, /*depth=*/2)) {
            if (first < 0)
              first = slice;
            last = slice;
          }
        }
        liveK.push_back(first < 0 ? std::pair<int64_t, int64_t>{0, 0}
                                  : std::pair<int64_t, int64_t>{first * bk, (last + 1) * bk});
      }
      if (llvm::all_of(liveK, [&](auto range) { return range == std::pair<int64_t, int64_t>{0, k}; }))
        liveK.clear();
    }
    if (debuggingParallelization()) {
      std::optional<DirectUses> cUses = directUses(c, product);
      llvm::errs() << "cleave-tile-matmuls-for-gpu: " << func.getName() << ": " << m << "x" << n << "x" << k << " by "
                   << bm << "," << bn << "," << bk << "," << tm << "," << tn << "," << split << ","
                   << tiling->vec << "," << tiling->db << ", "
                   << llvm::count(liveTiles, false) << " dead tiles of " << tilesM * tilesN << " (result "
                   << (cUses ? "read by loads only" : "used otherwise") << "), "
                   << (liveK.empty() ? "no" : "some") << " zero slices\n";
      if (!cUses) {
        llvm::errs() << "  result " << c << '\n';
        for (Operation *user : c.getUsers())
          llvm::errs() << "  result used by " << user->getName() << '\n';
      }
    }

    OpBuilder builder(product);
    Location loc = product.getLoc();
    auto constant = [&](OpBuilder &at, int64_t v) -> Value { return arith::ConstantIndexOp::create(at, loc, v); };
    // Workgroup memory, `[bk][rows / lanes + 1]` of `lanes`-wide vectors:
    // each thread reads its block's rows (and columns) `lanes` at a time,
    // 4 reads of workgroup memory for 64 multiply-adds instead of 16 (where
    // its block allows: `tm`, `tn` multiples of 4), its rows in groups of
    // `lanes` contiguous ones (`rowOf`). The padding puts the element a
    // thread stores into (a slice of a row-major A, `k` the fastest) on a
    // different bank from its neighbour's. Allocated in the kernel, in
    // SPIR-V's own storage class, the form `convert-gpu-to-spirv` turns into
    // workgroup variables: it handles neither a `gpu.func`'s workgroup
    // attributions nor `#gpu.address_space<workgroup>`.
    Attribute workgroup = spirv::StorageClassAttr::get(ctx, spirv::StorageClass::Workgroup);
    const int64_t aLanes = tm % 4 == 0 && (bk * bm / 4) % threads == 0 ? 4 : 1;
    const int64_t bLanes = tn % 4 == 0 && (bk * bn / 4) % threads == 0 ? 4 : 1;
    const bool doubled = tiling->db != 0;
    auto sharedType = [&](Type element, int64_t rows, int64_t lanes) {
      Type stored = lanes > 1 ? Type(VectorType::get({lanes}, element)) : element;
      SmallVector<int64_t> shape{bk, rows / lanes + 1};
      if (doubled)
        shape.insert(shape.begin(), 2);
      return MemRefType::get(shape, stored, MemRefLayoutAttrInterface{}, workgroup);
    };
    // An element of workgroup memory: in slice `buffer` (null when single).
    auto cell = [&](Value buffer, Value kk, Value group) -> SmallVector<Value> {
      if (buffer)
        return {buffer, kk, group};
      return {kk, group};
    };
    auto aShared = sharedType(aType.getElementType(), bm, aLanes);
    auto bShared = sharedType(bType.getElementType(), bn, bLanes);
    Value one = constant(builder, 1);
    Type cElement = cType.getElementType();
    Value partials;
    if (split > 1) {
      // Every element written before it's read (`cleave-mark-full-overwrites`
      // has run).
      partials = memref::AllocOp::create(builder, loc, MemRefType::get({split, m, n}, cElement));
      discardContents(builder, loc, partials, module);
    }
    // Device memory read 4 elements at a time (`tiling->vec`) along the
    // operand's contiguous dimension, its workgroup memory holding groups of
    // 4 rows: a `vector<4xf32>` view of its buffer, one load for 4 (a cast,
    // which `cleave-lower-gpu-launches` binds as the memory it views). Rows
    // contiguous (B of `x W`, both of `x^T u`): a load is a group, stored as
    // it came. `k` contiguous (A of `x W`, both of `x W^T`): a thread loads 4
    // rows' 4 `k`s and stores them transposed, 4 groups of 4 rows, which
    // takes its share of the slice in blocks of 16 (`quads`).
    auto rowsOf = [&](OpOperand *operand) { return operand == a ? bm : bn; };
    auto quads = [&](OpOperand *operand) {
      return product.getMatchingIndexingMap(operand).getDimPosition(1) == dimK;
    };
    auto vectorView = [&](OpOperand *operand, int64_t lanes) -> Value {
      if (!(tiling->vec & (quads(operand) ? 2 : 1)) || lanes != 4 ||
          (quads(operand) && (bk % 4 != 0 || (rowsOf(operand) * bk) % (16 * threads) != 0)))
        return {};
      auto type = cast<MemRefType>(operand->get().getType());
      if (type.getRank() != 2 || !type.getLayout().isIdentity() || !type.getElementType().isF32() ||
          type.getShape()[1] % 4 != 0)
        return {};
      auto viewType = MemRefType::get({type.getShape()[0], type.getShape()[1] / 4},
                                      VectorType::get({4}, type.getElementType()));
      return UnrealizedConversionCastOp::create(builder, loc, TypeRange{viewType}, ValueRange{operand->get()})
          .getResult(0);
    };
    Value aView = vectorView(a, aLanes), bView = vectorView(b, bLanes);
    auto launch = gpu::LaunchOp::create(builder, loc, constant(builder, n / bn), constant(builder, m / bm),
                                        constant(builder, split), constant(builder, threadsX),
                                        constant(builder, threadsY), one,
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
    // This thread's elements of the result: row `i`, column `j` of its block,
    // in groups of `lanes` contiguous ones, `threads * lanes` apart:
    // `lanes * (group * threads + t) + lane` within the tile.
    auto within = [&](OpBuilder &at, Value t, int64_t threadsAlong, int64_t lanes, int64_t i) {
      return add(at, mul(at, add(at, t, constant(at, (i / lanes) * threadsAlong)), lanes), constant(at, i % lanes));
    };
    auto rowOf = [&](OpBuilder &at, int64_t i) { return add(at, m0, within(at, ty, threadsY, aLanes, i)); };
    auto columnOf = [&](OpBuilder &at, int64_t j) { return add(at, n0, within(at, tx, threadsX, bLanes, j)); };
    SmallVector<Value> initial;
    Value zero = split > 1 ? arith::ConstantOp::create(kb, loc, kb.getZeroAttr(cElement)).getResult() : Value();
    for (int64_t i = 0; i < tm; ++i)
      for (int64_t j = 0; j < tn; ++j)
        initial.push_back(split > 1 ? zero
                          : filled  ? filled
                                    : memref::LoadOp::create(kb, loc, c,
                                                             indicesOf(product.getDpsInitOperand(0),
                                                                       {{dimM, rowOf(kb, i)}, {dimN, columnOf(kb, j)}}))
                                          .getResult());
    // This workgroup's slice of the reduction.
    Value kStart = mul(kb, launch.getBlockIds().z, slice);
    // A `rows` x `bk` slice of `operand` (its `outer` dimension `rows` long
    // from `origin`, `k` from `k0`), every thread a few groups of `lanes`
    // rows, neighbouring threads on the operand's contiguous dimension:
    // `slot` gives the thread's `r`-th group's place in the slice (`k`, the
    // group), `load` reads the thread's elements, `store` puts each group in
    // `shared[k][group]`.
    auto slot = [&](OpBuilder &at, OpOperand *operand, int64_t rows, int64_t lanes,
                    int64_t r) -> std::pair<Value, Value> {
      bool kFastest = product.getMatchingIndexingMap(operand).getDimPosition(1) == dimK;
      Value l = add(at, tid, constant(at, r * threads));
      int64_t fast = kFastest ? bk : rows / lanes;
      Value low = arith::RemUIOp::create(at, loc, l, constant(at, fast));
      Value high = arith::DivUIOp::create(at, loc, l, constant(at, fast));
      return kFastest ? std::make_pair(low, high) : std::make_pair(high, low);
    };
    auto load = [&](OpBuilder &at, OpOperand *operand, unsigned outer, Value origin, int64_t rows, int64_t lanes,
                    Value k0) {
      SmallVector<Value> values;
      Value view = operand == a ? aView : bView;
      if (view && quads(operand)) {
        // Block `t + i * threads`: rows `4 * (block / (bk / 4))`, `k`s
        // `4 * (block % (bk / 4))`, neighbouring threads along `k`; its 16
        // values by `k`, then row.
        for (int64_t i = 0; i < rows * bk / 16 / threads; ++i) {
          Value block = add(at, tid, constant(at, i * threads));
          Value kq = arith::RemUIOp::create(at, loc, block, constant(at, bk / 4));
          Value rq = arith::DivUIOp::create(at, loc, block, constant(at, bk / 4));
          Value quad = add(at, arith::DivUIOp::create(at, loc, k0, constant(at, 4)), kq);
          SmallVector<Value> reads;
          for (int64_t lane = 0; lane < 4; ++lane) {
            Value row = add(at, origin, add(at, mul(at, rq, 4), constant(at, lane)));
            reads.push_back(memref::LoadOp::create(at, loc, view, indicesOf(operand, {{outer, row}, {dimK, quad}})));
          }
          for (int64_t j = 0; j < 4; ++j)
            for (int64_t lane = 0; lane < 4; ++lane)
              values.push_back(vector::ExtractOp::create(at, loc, reads[lane], ArrayRef<int64_t>{j}).getResult());
        }
        return values;
      }
      for (int64_t r = 0; r < rows * bk / lanes / threads; ++r) {
        auto [kk, group] = slot(at, operand, rows, lanes, r);
        if (view) {
          // The group's 4 rows in one load from the view, taken apart.
          Value quad = add(at, arith::DivUIOp::create(at, loc, origin, constant(at, 4)), group);
          Value read = memref::LoadOp::create(at, loc, view, indicesOf(operand, {{outer, quad}, {dimK, add(at, k0, kk)}}));
          for (int64_t lane = 0; lane < lanes; ++lane)
            values.push_back(vector::ExtractOp::create(at, loc, read, ArrayRef<int64_t>{lane}).getResult());
          continue;
        }
        for (int64_t lane = 0; lane < lanes; ++lane) {
          Value row = add(at, mul(at, group, lanes), constant(at, lane));
          values.push_back(memref::LoadOp::create(
              at, loc, operand->get(),
              indicesOf(operand, {{outer, add(at, origin, row)}, {dimK, add(at, k0, kk)}})));
        }
      }
      return values;
    };
    auto store = [&](OpBuilder &at, OpOperand *operand, Value shared, Value buffer, int64_t rows, int64_t lanes,
                     ValueRange values) {
      if ((operand == a ? aView : bView) && quads(operand)) {
        // `load`'s blocks: each `k`'s 4 rows a group.
        Type element = values[0].getType();
        for (int64_t i = 0; i < rows * bk / 16 / threads; ++i) {
          Value block = add(at, tid, constant(at, i * threads));
          Value kq = arith::RemUIOp::create(at, loc, block, constant(at, bk / 4));
          Value rq = arith::DivUIOp::create(at, loc, block, constant(at, bk / 4));
          for (int64_t j = 0; j < 4; ++j) {
            Value group = vector::FromElementsOp::create(at, loc, VectorType::get({4}, element),
                                                         values.slice(i * 16 + j * 4, 4));
            memref::StoreOp::create(at, loc, group, shared, cell(buffer, add(at, mul(at, kq, 4), constant(at, j)), rq));
          }
        }
        return;
      }
      for (int64_t r = 0; r < rows * bk / lanes / threads; ++r) {
        auto [kk, group] = slot(at, operand, rows, lanes, r);
        ValueRange elements = values.slice(r * lanes, lanes);
        Value stored = lanes > 1 ? vector::FromElementsOp::create(
                                       at, loc, VectorType::get({lanes}, elements[0].getType()), elements)
                                       .getResult()
                                 : elements[0];
        memref::StoreOp::create(at, loc, stored, shared, cell(buffer, kk, group));
      }
    };
    // This thread's `count` values of a row of workgroup memory: its groups
    // (`group * threadsAlong + t`), each `lanes` wide, taken apart.
    auto readRow = [&](OpBuilder &at, Value shared, Value buffer, Value kk, Value t, int64_t threadsAlong,
                       int64_t lanes, int64_t count) {
      SmallVector<Value> values;
      for (int64_t group = 0; group < count / lanes; ++group) {
        Value read = memref::LoadOp::create(at, loc, shared,
                                            cell(buffer, kk, add(at, t, constant(at, group * threadsAlong))));
        for (int64_t lane = 0; lane < lanes; ++lane)
          values.push_back(lanes > 1 ? vector::ExtractOp::create(at, loc, read, ArrayRef<int64_t>{lane}).getResult()
                                     : read);
      }
      return values;
    };
    const int64_t aCount = bm * bk / threads, bCount = bn * bk / threads;
    Value aArg = body->getArgument(a->getOperandNumber()), bArg = body->getArgument(b->getOperandNumber()),
          cArg = body->getArgument(2);
    Value yielded = body->getTerminator()->getOperand(0);
    // Double-buffered through registers: the slice at `k0` came in with the
    // loop (loaded during the previous slice's arithmetic, the first before
    // the loop); stored to workgroup memory, it's the next slice that's
    // loaded while this one is computed, the loads' latency hidden behind
    // the multiply-adds. The last iteration loads its own slice again, never
    // past the reduction's end.
    // Within this row of tiles' live slices (`liveK`, by `y`): from `kFirst`
    // to `kEnd`, the first slice loaded from no further than the last one
    // (the range may be empty).
    Value kFirst = kStart, kEnd = add(kb, kStart, constant(kb, slice));
    if (!liveK.empty()) {
      auto byRow = [&](auto pick) {
        Value value = constant(kb, pick(liveK.back()));
        for (int64_t tm = tilesM - 2; tm >= 0; --tm) {
          Value here = arith::CmpIOp::create(kb, loc, arith::CmpIPredicate::eq, launch.getBlockIds().y,
                                             constant(kb, tm));
          value = arith::SelectOp::create(kb, loc, here, constant(kb, pick(liveK[tm])), value);
        }
        return value;
      };
      kFirst = arith::MaxUIOp::create(kb, loc, kStart, byRow([](auto range) { return range.first; }));
      kEnd = arith::MinUIOp::create(kb, loc, kEnd, byRow([](auto range) { return range.second; }));
    }
    Value kLoad = liveK.empty() ? kFirst : arith::MinUIOp::create(kb, loc, kFirst, constant(kb, k - bk)).getResult();
    const size_t accumulators = initial.size();
    // The slice's `bk` steps unrolled, read from workgroup memory `buffer`:
    // straight-line code the driver's compiler schedules as a whole, no loop
    // carrying `tm * tn` values.
    auto compute = [&](OpBuilder &lb, Value buffer, SmallVector<Value> &acc) {
      for (int64_t step = 0; step < bk; ++step) {
        Value kk = constant(lb, step);
        SmallVector<Value> aValues = readRow(lb, as, buffer, kk, ty, threadsY, aLanes, tm);
        SmallVector<Value> bValues = readRow(lb, bs, buffer, kk, tx, threadsX, bLanes, tn);
        for (int64_t i = 0; i < tm; ++i)
          for (int64_t j = 0; j < tn; ++j) {
            IRMapping mapping;
            mapping.map(aArg, aValues[i]);
            mapping.map(bArg, bValues[j]);
            mapping.map(cArg, acc[i * tn + j]);
            for (Operation &inner : body->without_terminator())
              lb.clone(inner, mapping);
            acc[i * tn + j] = mapping.lookupOrDefault(yielded);
          }
      }
    };
    // The slice after `k0`'s, never past the reduction's end (the last
    // iteration loads its own again).
    auto nextOf = [&](OpBuilder &lb, Value k0) -> Value {
      return arith::MinUIOp::create(lb, loc, add(lb, k0, constant(lb, bk)),
                                    arith::SubIOp::create(lb, loc, kEnd, constant(lb, bk)));
    };
    scf::ForOp outerLoop;
    if (doubled) {
      // Two slices in workgroup memory (`tiling->db`), one barrier a slice:
      // the first stored before the loop, the second loaded into registers;
      // each iteration stores the slice it came with into the other half,
      // loads the one after into registers (carried by the loop, so that the
      // driver's compiler can't sink the loads past the arithmetic hiding
      // their latency), computes with the current half, and waits once:
      // every thread's stores seen before the next iteration reads them,
      // its reads of this half done before the iteration after overwrites it.
      Value zero = constant(kb, 0);
      store(kb, a, as, zero, bm, aLanes, load(kb, a, dimM, m0, bm, aLanes, kLoad));
      store(kb, b, bs, zero, bn, bLanes, load(kb, b, dimN, n0, bn, bLanes, kLoad));
      Value secondLoad = arith::MinUIOp::create(kb, loc, add(kb, kLoad, constant(kb, bk)), constant(kb, k - bk));
      SmallVector<Value> carried(initial);
      for (Value v : load(kb, a, dimM, m0, bm, aLanes, secondLoad))
        carried.push_back(v);
      for (Value v : load(kb, b, dimN, n0, bn, bLanes, secondLoad))
        carried.push_back(v);
      gpu::BarrierOp::create(kb, loc);
      outerLoop = scf::ForOp::create(
          kb, loc, kFirst, kEnd, constant(kb, bk), carried,
          [&](OpBuilder &lb, Location, Value k0, ValueRange inflow) {
            ValueRange accumulated = inflow.take_front(accumulators);
            ValueRange aSlice = inflow.slice(accumulators, aCount), bSlice = inflow.drop_front(accumulators + aCount);
            Value index = arith::DivUIOp::create(lb, loc, arith::SubIOp::create(lb, loc, k0, kFirst), constant(lb, bk));
            Value current = arith::RemUIOp::create(lb, loc, index, constant(lb, 2));
            Value other = arith::SubIOp::create(lb, loc, constant(lb, 1), current);
            store(lb, a, as, other, bm, aLanes, aSlice);
            store(lb, b, bs, other, bn, bLanes, bSlice);
            Value after = nextOf(lb, nextOf(lb, k0));
            SmallVector<Value> aAfter = load(lb, a, dimM, m0, bm, aLanes, after),
                               bAfter = load(lb, b, dimN, n0, bn, bLanes, after);
            SmallVector<Value> acc(accumulated.begin(), accumulated.end());
            compute(lb, current, acc);
            gpu::BarrierOp::create(lb, loc);
            acc.append(aAfter.begin(), aAfter.end());
            acc.append(bAfter.begin(), bAfter.end());
            scf::YieldOp::create(lb, loc, acc);
          });
    } else {
      // Double-buffered through registers (above).
      SmallVector<Value> carried(initial);
      for (Value v : load(kb, a, dimM, m0, bm, aLanes, kLoad))
        carried.push_back(v);
      for (Value v : load(kb, b, dimN, n0, bn, bLanes, kLoad))
        carried.push_back(v);
      outerLoop = scf::ForOp::create(
          kb, loc, kFirst, kEnd, constant(kb, bk), carried,
          [&](OpBuilder &lb, Location, Value k0, ValueRange inflow) {
            ValueRange accumulated = inflow.take_front(accumulators);
            ValueRange aSlice = inflow.slice(accumulators, aCount), bSlice = inflow.drop_front(accumulators + aCount);
            store(lb, a, as, Value(), bm, aLanes, aSlice);
            store(lb, b, bs, Value(), bn, bLanes, bSlice);
            gpu::BarrierOp::create(lb, loc);
            Value next = nextOf(lb, k0);
            SmallVector<Value> aNext = load(lb, a, dimM, m0, bm, aLanes, next),
                               bNext = load(lb, b, dimN, n0, bn, bLanes, next);
            SmallVector<Value> acc(accumulated.begin(), accumulated.end());
            compute(lb, Value(), acc);
            gpu::BarrierOp::create(lb, loc);
            acc.append(aNext.begin(), aNext.end());
            acc.append(bNext.begin(), bNext.end());
            scf::YieldOp::create(lb, loc, acc);
          });
    }
    for (int64_t i = 0; i < tm; ++i)
      for (int64_t j = 0; j < tn; ++j) {
        if (split > 1)
          memref::StoreOp::create(kb, loc, outerLoop.getResult(i * tn + j), partials,
                                  ValueRange{launch.getBlockIds().z, rowOf(kb, i), columnOf(kb, j)});
        else
          memref::StoreOp::create(
              kb, loc, outerLoop.getResult(i * tn + j), c,
              indicesOf(product.getDpsInitOperand(0), {{dimM, rowOf(kb, i)}, {dimN, columnOf(kb, j)}}));
      }
    if (!liveTiles.empty()) {
      // The workgroup's work done if its tile is live: bit `y * tilesN + x`
      // of a mask in 32-bit words (`liveTiles`), the same for the whole
      // workgroup (its barriers then reached by all of it or none).
      SmallVector<Operation *> work;
      for (Operation &op : kernel.without_terminator())
        if (!isa<memref::AllocOp>(op))
          work.push_back(&op);
      Type i32 = kb.getI32Type();
      Value tile = add(kb, mul(kb, launch.getBlockIds().y, tilesN), launch.getBlockIds().x);
      Value word = arith::IndexCastOp::create(kb, loc, i32, arith::DivUIOp::create(kb, loc, tile, constant(kb, 32)));
      Value bit = arith::IndexCastOp::create(kb, loc, i32, arith::RemUIOp::create(kb, loc, tile, constant(kb, 32)));
      auto maskOf = [&](size_t w) {
        uint32_t mask = 0;
        for (size_t t = w * 32; t < std::min(liveTiles.size(), (w + 1) * 32); ++t)
          if (liveTiles[t])
            mask |= uint32_t(1) << (t % 32);
        return arith::ConstantIntOp::create(kb, loc, i32, int64_t(int32_t(mask))).getResult();
      };
      size_t words = (liveTiles.size() + 31) / 32;
      Value mask = maskOf(words - 1);
      for (size_t w = words - 1; w-- > 0;)
        mask = arith::SelectOp::create(
            kb, loc,
            arith::CmpIOp::create(kb, loc, arith::CmpIPredicate::eq, word,
                                  arith::ConstantIntOp::create(kb, loc, i32, int64_t(w))),
            maskOf(w), mask);
      Value one32 = arith::ConstantIntOp::create(kb, loc, i32, 1);
      Value live = arith::CmpIOp::create(
          kb, loc, arith::CmpIPredicate::ne,
          arith::AndIOp::create(kb, loc, arith::ShRUIOp::create(kb, loc, mask, bit), one32),
          arith::ConstantIntOp::create(kb, loc, i32, 0));
      auto guard = scf::IfOp::create(kb, loc, TypeRange{}, live, /*withElseRegion=*/false);
      Operation *end = guard.thenBlock()->getTerminator();
      for (Operation *op : work)
        op->moveBefore(end);
    }
    if (split > 1) {
      // The slices added to the result, an element per thread, in order.
      constexpr int64_t kThreads = 256;
      int64_t elements = m * n;
      OpBuilder after(product);
      auto sum = gpu::LaunchOp::create(after, loc, constant(after, (elements + kThreads - 1) / kThreads),
                                       constant(after, 1), constant(after, 1), constant(after, kThreads),
                                       constant(after, 1), constant(after, 1),
                                       /*dynamicSharedMemorySize=*/nullptr, /*asyncTokenType=*/nullptr);
      Block &sumBody = sum.getBody().front();
      if (sumBody.empty() || !sumBody.back().hasTrait<OpTrait::IsTerminator>()) {
        OpBuilder atEnd = OpBuilder::atBlockEnd(&sumBody);
        gpu::TerminatorOp::create(atEnd, loc);
      }
      OpBuilder sb(&sumBody, std::prev(sumBody.end()));
      Value index = add(sb, mul(sb, sum.getBlockIds().x, kThreads), sum.getThreadIds().x);
      Value inside = arith::CmpIOp::create(sb, loc, arith::CmpIPredicate::ult, index, constant(sb, elements));
      auto guard = scf::IfOp::create(sb, loc, TypeRange{}, inside, /*withElseRegion=*/false);
      OpBuilder gb = OpBuilder::atBlockBegin(guard.thenBlock());
      Value row = arith::DivUIOp::create(gb, loc, index, constant(gb, n));
      Value column = arith::RemUIOp::create(gb, loc, index, constant(gb, n));
      SmallVector<Value> at = indicesOf(product.getDpsInitOperand(0), {{dimM, row}, {dimN, column}});
      Value total = filled ? filled : memref::LoadOp::create(gb, loc, c, at).getResult();
      Value x = additive->getOperand(0) == cArgument ? additive->getOperand(1) : additive->getOperand(0);
      for (int64_t z = 0; z < split; ++z) {
        Value part = memref::LoadOp::create(gb, loc, partials, ValueRange{constant(gb, z), row, column});
        IRMapping mapping;
        mapping.map(cArgument, total);
        mapping.map(x, part);
        total = gb.clone(*additive, mapping)->getResult(0);
      }
      memref::StoreOp::create(gb, loc, total, c, at);
      memref::DeallocOp::create(after, loc, partials);
    }
    product->erase();
    if (fill)
      fill->erase();
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
  // `aliases`: other memrefs of the same memory, whose uses count too;
  // `escapes`: operations that hand it elsewhere, uses as well.
  auto discardBeforeFirstUse = [&](Value freshMemref, Block *block, ArrayRef<Value> aliases = {},
                                   ArrayRef<Operation *> escapes = {}) {
    auto type = dyn_cast<MemRefType>(freshMemref.getType());
    if (!type || !type.hasStaticShape())
      return;
    // The first use, through views, as the operation of `block` that holds it.
    Operation *first = nullptr;
    SmallVector<Value> pending{freshMemref};
    pending.append(aliases.begin(), aliases.end());
    for (Operation *escape : escapes)
      if (Operation *holder = block->findAncestorOpInBlock(*escape))
        if (!first || holder->isBeforeInBlock(first))
          first = holder;
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
  // A fresh heap allocation made a memref in the function itself (a struct's
  // tensor field a kernel writes: `accumulate`'s sums, uploaded undefined).
  module.walk([&](UnrealizedConversionCastOp cast) {
    SmallVector<Value> aliases;
    SmallVector<Operation *> escapes;
    if (!cast->getParentOfType<gpu::GPUModuleOp>() && isFreshDescriptor(cast, &aliases, &escapes))
      discardBeforeFirstUse(cast->getResult(0), cast->getBlock(), aliases, escapes);
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

// The most a host loop's folded iterations may hold at once
// (`foldLoopIntoGrids`, each iteration's allocations at their peak): about
// what the device's last-level cache holds, which Vulkan doesn't tell.
// 64 MiB by default, measured on an RTX 4070 SUPER (48 MiB of L2) with
// nanoLM's attention, 12 heads: forward, 8 MiB a head, 17.4 ms unfolded,
// 11.6 by 4 heads, 11.3 by 6, 15.6 all 12 at once; backward, 12 MiB a head,
// 24.0 ms unfolded, 19.5 by 2, 16.5 by 4, 19.1 by 6 (half of the scores,
// never read, never written). `CLEAVE_GPU_FOLD_BYTES` at compile time to fit
// another device.
static int64_t foldedBytes() {
  static const int64_t bytes = [] {
    const char *v = std::getenv("CLEAVE_GPU_FOLD_BYTES");
    return v ? std::strtoll(v, nullptr, 10) : int64_t(64) << 20;
  }();
  return bytes;
}

// Folds `loop` into the grids of the kernels it launches
// (`cleaveFoldLoopsIntoGrids`), if it can.
static bool foldLoopIntoGrids(scf::ForOp loop, ModuleOp module, FreshMemo &fresh) {
  auto refuse = [&](const char *why) {
    if (debuggingParallelization()) {
      auto func = loop->getParentOfType<func::FuncOp>();
      llvm::errs() << "cleave-fold-loops-into-grids: " << (func ? func.getName() : StringRef("?")) << ": " << why
                   << '\n';
    }
    return false;
  };
  std::optional<int64_t> lower = getConstantIntValue(loop.getLowerBound()),
                         upper = getConstantIntValue(loop.getUpperBound()), step = getConstantIntValue(loop.getStep());
  if (!lower || !upper || !step || *step <= 0)
    return false;
  const int64_t trips = (*upper - *lower + *step - 1) / *step;
  if (trips < 2)
    return false;
  // The body: the launches, the allocations they use (static shapes, only
  // their accesses using them besides the body's deallocations and
  // `cleave_vk_discard`s), and the rest, which stays in the loop, and is
  // checked below never to touch what the launches do (`residue`).
  SmallVector<gpu::LaunchOp> launches;
  for (Operation &op : loop.getBody()->without_terminator())
    if (auto launch = dyn_cast<gpu::LaunchOp>(op))
      launches.push_back(launch);
  if (launches.empty())
    return false;
  auto inLaunch = [&](Operation *op) {
    return llvm::any_of(launches, [&](gpu::LaunchOp launch) { return launch->isAncestor(op); });
  };
  llvm::SetVector<Operation *> allocations; // `memref.alloc`s
  for (Operation &op : loop.getBody()->without_terminator())
    if (auto alloc = dyn_cast<memref::AllocOp>(op);
        alloc && llvm::any_of(alloc->getUsers(), [&](Operation *user) { return inLaunch(user); })) {
      if (!alloc.getType().hasStaticShape() || !alloc.getType().getLayout().isIdentity())
        return refuse("allocates a buffer of a shape known at run time");
      allocations.insert(alloc);
    }
  // The discards of those allocations: their first address's.
  llvm::DenseSet<Operation *> discarded;
  SmallVector<Operation *> discards;
  for (Operation &op : loop.getBody()->without_terminator()) {
    auto call = dyn_cast<func::CallOp>(op);
    if (!call || call.getCallee() != "cleave_vk_discard")
      continue;
    Value at = call.getOperand(0);
    while (Operation *def = at.getDefiningOp()) {
      if (auto pointer = dyn_cast<memref::ExtractAlignedPointerAsIndexOp>(def)) {
        at = pointer.getSource();
        break;
      }
      if (!isa<arith::IndexCastOp, arith::IndexCastUIOp, arith::AddIOp>(def))
        break;
      at = def->getOperand(0);
    }
    if (Operation *alloc = at.getDefiningOp(); alloc && allocations.contains(alloc)) {
      discarded.insert(alloc);
      discards.push_back(call);
    }
  }
  // The iterations folded together, `group` of them per launch: what one
  // iteration's kernels pass each other through the device's cache would
  // otherwise go through its memory (nanoLM's 12 heads at once, 8 MiB of
  // scores and weights each: 50% slower folded). As many as the largest
  // divisor of the trip count whose allocations fit `foldedBytes`, the rest
  // a host loop over the groups.
  // An iteration's footprint: the most bytes its allocations hold at once, in
  // the body's order (attention's gradient frees its scores before it
  // allocates its products' partials).
  int64_t bytes = 0;
  {
    int64_t live = 0;
    auto size = [](Operation *op) {
      MemRefType type = cast<memref::AllocOp>(op).getType();
      return type.getNumElements() * ((type.getElementTypeBitWidth() + 7) / 8);
    };
    for (Operation &op : loop.getBody()->without_terminator()) {
      if (allocations.contains(&op))
        live += size(&op);
      else if (auto dealloc = dyn_cast<memref::DeallocOp>(op);
               dealloc && allocations.contains(dealloc.getMemref().getDefiningOp()))
        live -= size(dealloc.getMemref().getDefiningOp());
      bytes = std::max(bytes, live);
    }
  }
  if (debuggingParallelization())
    llvm::errs() << "cleave-fold-loops-into-grids: an iteration holds " << bytes << " bytes at most\n";
  int64_t group = trips;
  while (group > 1 && (trips % group != 0 || bytes * group > foldedBytes()))
    --group;
  if (group < 2)
    return refuse("two iterations' allocations together don't fit a cache");
  for (Operation *alloc : allocations)
    for (Operation *user : alloc->getUsers()) {
      if (isa<memref::DeallocOp>(user) && user->getBlock() == loop.getBody())
        continue;
      if (auto pointer = dyn_cast<memref::ExtractAlignedPointerAsIndexOp>(user); pointer && !inLaunch(user))
        continue;
      if (inLaunch(user) && isa<memref::LoadOp, memref::StoreOp, vector::LoadOp, vector::StoreOp>(user))
        continue;
      return refuse("uses an allocation otherwise than by the kernels' accesses");
    }
  // The launches: values the body computes sunk into them, so that they
  // take only its allocations and its induction variable from it; grids
  // whose `z` the iterations can multiply.
  for (gpu::LaunchOp launch : launches) {
    // The pure computations of the body a launch takes the results of,
    // cloned at its start, in their order (MLIR's
    // `sinkOperationsIntoLaunchOp` leaves those of the induction variable).
    llvm::SetVector<Operation *> computations;
    std::function<bool(Value)> computing = [&](Value v) {
      Operation *def = v.getDefiningOp();
      if (!def || def->getBlock() != loop.getBody() || allocations.contains(def) || computations.contains(def))
        return true;
      if (def->getNumRegions() != 0 || !isMemoryEffectFree(def) || !llvm::all_of(def->getOperands(), computing))
        return false;
      computations.insert(def);
      return true;
    };
    llvm::SetVector<Value> taken;
    getUsedValuesDefinedAbove(launch.getBody(), taken);
    if (llvm::all_of(taken, computing)) {
      OpBuilder start = OpBuilder::atBlockBegin(&launch.getBody().front());
      IRMapping clones;
      for (Operation *computation : computations)
        start.clone(*computation, clones);
      for (Operation *computation : computations)
        for (Value result : computation->getResults())
          result.replaceUsesWithIf(clones.lookup(result),
                                   [&](OpOperand &use) { return launch->isAncestor(use.getOwner()); });
    }
    for (Value operand : launch->getOperands())
      if (Operation *def = operand.getDefiningOp(); def && loop->isAncestor(def))
        return refuse("a launch's size the body computes");
    std::optional<int64_t> z = getConstantIntValue(launch.getGridSizeZ());
    constexpr int64_t kMostWorkgroupsAlongZ = 65535; // Vulkan's guaranteed `maxComputeWorkGroupCount`
    if (!z || *z * group > kMostWorkgroupsAlongZ || launch.hasClusterSize() || !launch.getAsyncDependencies().empty())
      return refuse("a launch whose grid can't take the iterations");
    llvm::SetVector<Value> captured;
    getUsedValuesDefinedAbove(launch.getBody(), captured);
    for (Value v : captured) {
      Operation *at = v.getDefiningOp() ? v.getDefiningOp() : v.getParentBlock()->getParentOp();
      if (loop->isAncestor(at) && v != loop.getInductionVar() && !allocations.contains(v.getDefiningOp())) {
        if (debuggingParallelization())
          llvm::errs() << "  " << v << '\n';
        return refuse("a launch takes a value the body computes");
      }
    }
  }
  // What the launches write: their own memory (workgroup memory), an
  // allocation of the body's, or else one of the memrefs they take directly.
  // Those are fresh allocations no other memref the launches take views
  // (`isFreshUnaliased`, distinct roots), that only the kernels' loads and
  // stores touch in the loop, where no iteration touches what another writes
  // (`accessesAreDisjoint`).
  llvm::SetVector<Value> written, taken;
  for (gpu::LaunchOp launch : launches) {
    llvm::SetVector<Value> captured;
    getUsedValuesDefinedAbove(launch.getBody(), captured);
    for (Value v : captured)
      if (isa<BaseMemRefType>(v.getType()) && !allocations.contains(v.getDefiningOp()))
        taken.insert(v);
    WalkResult walked = launch.getBody().walk([&](Operation *inner) {
      if (isa<gpu::BarrierOp, spirv::ControlBarrierOp>(inner))
        return WalkResult::advance();
      auto effects = dyn_cast<MemoryEffectOpInterface>(inner);
      if (!effects)
        return inner->hasTrait<OpTrait::HasRecursiveMemoryEffects>() ||
                       (inner->getNumRegions() == 0 && isMemoryEffectFree(inner))
                   ? WalkResult::advance()
                   : WalkResult::interrupt();
      SmallVector<MemoryEffects::EffectInstance> instances;
      effects.getEffects(instances);
      for (const MemoryEffects::EffectInstance &effect : instances) {
        if (!isa<MemoryEffects::Write>(effect.getEffect()))
          continue;
        Value target = effect.getValue();
        if (!target)
          return WalkResult::interrupt();
        Operation *at = target.getDefiningOp() ? target.getDefiningOp() : target.getParentBlock()->getParentOp();
        if (launch->isAncestor(at) || allocations.contains(target.getDefiningOp()))
          continue;
        if (!isa<memref::StoreOp>(inner) || !taken.contains(target))
          return WalkResult::interrupt();
        written.insert(target);
      }
      return WalkResult::advance();
    });
    if (walked.wasInterrupted())
      return refuse("a kernel writes memory it can't be told it owns");
  }
  // A fresh descriptor's other names (`isFreshDescriptor`'s aliases: the
  // function converts it to a memref again where it reads the field) and
  // whatever else takes it: nowhere in the loop but for an address.
  auto freshHere = [&](Value memref) {
    if (isFreshUnaliased(memref, module, fresh))
      return true;
    auto cast = memref.getDefiningOp<UnrealizedConversionCastOp>();
    SmallVector<Value> aliases;
    SmallVector<Operation *> escapes;
    if (!cast || !isFreshDescriptor(cast, &aliases, &escapes, /*arena=*/true))
      return false;
    if (llvm::any_of(escapes, [&](Operation *op) { return loop->isAncestor(op); }))
      return false;
    return llvm::all_of(aliases, [&](Value alias) {
      return llvm::all_of(alias.getUsers(), [&](Operation *user) {
        return !loop->isAncestor(user) ||
               isa<memref::ExtractStridedMetadataOp, memref::ExtractAlignedPointerAsIndexOp>(user);
      });
    });
  };
  for (Value memref : written) {
    if (!freshHere(memref)) {
      if (debuggingParallelization())
        llvm::errs() << "  " << memref << '\n';
      return refuse("writes a buffer that isn't a fresh allocation");
    }
    for (Value other : taken)
      if (other != memref && viewRoot(other) == viewRoot(memref))
        return refuse("writes a buffer it also takes a view of");
    // Its address or shape taken (the deallocation's bookkeeping) reaches
    // no element.
    for (Operation *user : memref.getUsers())
      if (loop->isAncestor(user) && !isa<memref::LoadOp, memref::StoreOp, memref::ExtractStridedMetadataOp,
                                         memref::ExtractAlignedPointerAsIndexOp>(user))
        return refuse("uses a buffer it writes otherwise than by loads and stores");
    if (!accessesAreDisjoint(loop, {loop.getInductionVar()}, memref))
      return refuse("two iterations may touch an element of a buffer it writes");
  }

  // The rest of the body, which stays in the loop, run after every launch
  // (`residue`): what it reads or writes isn't what the launches take or
  // allocate (its own small buffers: a deallocation's bookkeeping, its
  // `dealloc_helper`), and it calls nothing but the runtime's regions and
  // that helper. Freeing what the launches use is allowed: in the loop's own
  // order an iteration's launches would then have used freed memory, so the
  // program never does it before the last of them, and they all now come
  // first.
  llvm::DenseSet<Value> touched;
  for (Value v : taken)
    touched.insert(viewRoot(v));
  for (Operation *alloc : allocations)
    touched.insert(alloc->getResult(0));
  auto ours = [&](Operation *op) {
    if (isa<gpu::LaunchOp>(op) || allocations.contains(op) || llvm::is_contained(discards, op))
      return true;
    auto dealloc = dyn_cast<memref::DeallocOp>(op);
    return dealloc && allocations.contains(dealloc.getMemref().getDefiningOp());
  };
  for (Operation &op : loop.getBody()->without_terminator()) {
    if (ours(&op))
      continue;
    WalkResult walked = op.walk([&](Operation *inner) {
      if (auto call = dyn_cast<func::CallOp>(inner)) {
        StringRef callee = call.getCallee();
        if (callee != "cleave_region_enter" && callee != "cleave_region_exit" && callee != "dealloc_helper" &&
            callee != "cleave_vk_discard")
          return WalkResult::interrupt();
        for (Value operand : call.getOperands())
          if (isa<BaseMemRefType>(operand.getType()) && touched.contains(viewRoot(operand)))
            return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (inner->hasTrait<OpTrait::HasRecursiveMemoryEffects>() || isMemoryEffectFree(inner))
        return WalkResult::advance();
      auto effects = dyn_cast<MemoryEffectOpInterface>(inner);
      if (!effects)
        return WalkResult::interrupt();
      SmallVector<MemoryEffects::EffectInstance> instances;
      effects.getEffects(instances);
      for (const MemoryEffects::EffectInstance &effect : instances) {
        Value target = effect.getValue();
        if (!target)
          return WalkResult::interrupt();
        if (touched.contains(viewRoot(target)) && !isa<MemoryEffects::Free>(effect.getEffect()))
          return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (walked.wasInterrupted())
      return refuse("the rest of the body touches what the launches do");
  }

  // Folded: the allocations, for a group's iterations, before the loop; the
  // launches moved before it in their order (into a loop over the groups,
  // `first` its group's first iteration, when there are several), each over
  // a group's iterations; the allocations freed after it; the loop erased.
  Location loc = loop.getLoc();
  OpBuilder builder(loop);
  llvm::DenseMap<Operation *, Value> expanded;
  for (Operation *op : allocations) {
    auto alloc = cast<memref::AllocOp>(op);
    MemRefType type = alloc.getType();
    SmallVector<int64_t> shape{group};
    llvm::append_range(shape, type.getShape());
    auto all = memref::AllocOp::create(
        builder, loc, MemRefType::get(shape, type.getElementType(), MemRefLayoutAttrInterface{}, type.getMemorySpace()),
        alloc.getAlignmentAttr());
    expanded[op] = all;
    if (discarded.contains(op)) {
      auto [first, bytes] = memrefSpan(builder, loc, all);
      func::CallOp::create(builder, loc, "cleave_vk_discard", TypeRange{}, ValueRange{first, bytes});
    }
  }
  if (debuggingParallelization())
    llvm::errs() << "cleave-fold-loops-into-grids: " << loop->getParentOfType<func::FuncOp>().getName() << ": folded "
                 << trips << " iterations by " << group << '\n';
  Operation *place = loop;
  Value first;
  if (group < trips) {
    auto groups = scf::ForOp::create(builder, loc, arith::ConstantIndexOp::create(builder, loc, 0),
                                     arith::ConstantIndexOp::create(builder, loc, trips / group),
                                     arith::ConstantIndexOp::create(builder, loc, 1));
    place = groups.getBody()->getTerminator();
    OpBuilder at(place);
    first = arith::MulIOp::create(at, loc, groups.getInductionVar(), arith::ConstantIndexOp::create(at, loc, group));
  }
  for (gpu::LaunchOp launch : launches) {
    launch->moveBefore(place);
    OpBuilder outside(launch);
    const int64_t z = *getConstantIntValue(launch.getGridSizeZ());
    launch.getGridSizeZMutable().assign(arith::ConstantIndexOp::create(outside, loc, z * group));
    Block &body = launch.getBody().front();
    OpBuilder inside = OpBuilder::atBlockBegin(&body);
    // The iteration `z / gz`, the kernel's own `z` the rest.
    Value blockZ = launch.getBlockIds().z;
    Value zSize = arith::ConstantIndexOp::create(inside, loc, z);
    auto own = arith::RemUIOp::create(inside, loc, blockZ, zSize);
    auto iteration = arith::DivUIOp::create(inside, loc, blockZ, zSize);
    blockZ.replaceAllUsesExcept(own, SmallPtrSet<Operation *, 2>{own, iteration});
    launch.getGridSize().z.replaceAllUsesWith(zSize);
    Value index = first ? arith::AddIOp::create(inside, loc, first, iteration).getResult() : iteration.getResult();
    Value iv = arith::AddIOp::create(
        inside, loc, arith::ConstantIndexOp::create(inside, loc, *lower),
        arith::MulIOp::create(inside, loc, index, arith::ConstantIndexOp::create(inside, loc, *step)));
    if (!loop.getInductionVar().getType().isIndex())
      iv = arith::IndexCastOp::create(inside, loc, loop.getInductionVar().getType(), iv);
    loop.getInductionVar().replaceUsesWithIf(iv, [&](OpOperand &use) { return launch->isAncestor(use.getOwner()); });
    // Each access to an allocation, to this iteration's slice of it: the
    // iteration its first index.
    for (auto [op, all] : expanded)
      for (Operation *user : llvm::make_early_inc_range(op->getUsers())) {
        if (!launch->isAncestor(user))
          continue;
        OpBuilder at(user);
        auto indices = [&](ValueRange old) {
          SmallVector<Value> result{iteration.getResult()};
          llvm::append_range(result, old);
          return result;
        };
        if (auto load = dyn_cast<memref::LoadOp>(user))
          load.replaceAllUsesWith(memref::LoadOp::create(at, load.getLoc(), all, indices(load.getIndices())).getResult());
        else if (auto store = dyn_cast<memref::StoreOp>(user))
          memref::StoreOp::create(at, store.getLoc(), store.getValueToStore(), all, indices(store.getIndices()));
        else if (auto load = dyn_cast<vector::LoadOp>(user))
          load.replaceAllUsesWith(
              vector::LoadOp::create(at, load.getLoc(), load.getVectorType(), all, indices(load.getIndices()))
                  .getResult());
        else if (auto store = dyn_cast<vector::StoreOp>(user))
          vector::StoreOp::create(at, store.getLoc(), store.getValueToStore(), all, indices(store.getIndices()));
        user->erase();
      }
  }
  builder.setInsertionPointAfter(loop);
  for (auto [op, all] : expanded)
    memref::DeallocOp::create(builder, loc, all);
  // The allocations' deallocations and discards gone, with what computed
  // the addresses for them; the loop too when nothing else remains in it but
  // pure operations and the runtime's regions.
  std::function<void(Operation *)> eraseUnused = [&](Operation *op) {
    SmallVector<Operation *> operands;
    for (Value operand : op->getOperands())
      if (Operation *def = operand.getDefiningOp())
        operands.push_back(def);
    op->erase();
    for (Operation *def : operands)
      if (def->use_empty() && isMemoryEffectFree(def) && def->getBlock() == loop.getBody())
        eraseUnused(def);
  };
  for (Operation *call : discards)
    eraseUnused(call);
  for (Operation *op : allocations) {
    for (Operation *user : llvm::make_early_inc_range(op->getUsers()))
      if (isa<memref::DeallocOp>(user))
        user->erase();
    for (Operation *user : llvm::make_early_inc_range(op->getUsers()))
      if (user->use_empty() && isMemoryEffectFree(user))
        user->erase();
    op->erase();
  }
  bool residue = loop.getNumRegionIterArgs() != 0 ||
                 llvm::any_of(loop.getBody()->without_terminator(), [](Operation &op) {
                   auto call = dyn_cast<func::CallOp>(op);
                   bool region = call && (call.getCallee() == "cleave_region_enter" ||
                                          call.getCallee() == "cleave_region_exit");
                   return !region && !isMemoryEffectFree(&op);
                 });
  if (!residue)
    loop.erase();
  return true;
}

// Vulkan backend: a host loop launching kernels (`causal_attention`'s loop
// over heads: per head, its scores' product, their softmax, the product by
// the values, each a kernel over the head's slices) folded into the
// kernels: each launched once, the loop's iterations more workgroups along
// `z` (`gz * trips`: the iteration `z / gz`, the kernel's own `z` the rest),
// the launches in the loop's order. What the loop allocates for an
// iteration (the scores) allocated once for all, a slice each. When its
// iterations are independent: each kernel then runs every iteration's
// workgroups in any order, after the previous kernel's, with the same
// values: 4 heads of a 128-wide attention made 24 dispatches of a few
// workgroups each, 3 times longer than the 6 they are. By groups of
// iterations when all of them wouldn't fit the device's cache together
// (`foldedBytes`: each one's kernels pass each other its scores through it),
// a host loop over the groups left: nanoLM's 12 heads by 6, 35% faster. The loop
// counts from a constant to a constant; its body launches,
// allocates (static shapes the kernels alone access, with their
// `cleave_vk_discard`s), frees, computes without effects and calls
// `cleave_region_enter`/`exit` in balanced pairs, nothing else; what the
// kernels write beyond their own memory and the loop's allocations is fresh
// memory they access directly, no other memref they take a view of it, and
// no iteration touches what another writes (`accessesAreDisjoint`, over the
// kernels' block and thread ids too). Innermost loops first, so a nest
// folds level by level. Returns how many loops it folded.
// A host `scf.while` counting `iv` from `lb` by 1 while `iv < ub` (signed,
// `ub` defined outside it) made an `scf.for`, whatever else its condition's
// region does: `UpliftWhileToFor` wants that region to only compare, but
// ownership-based deallocation puts its bookkeeping there for what the loop
// carries (`dealloc_helper`, flags in the loop's values: attention's
// gradient loop over heads). That region runs once more than the body, the
// last time when `iv` reaches `max(lb, ub)`: in the `scf.for`, it runs at
// the start of each iteration, then once after the loop, which gives the
// `scf.while`'s results. Whether it rewrote `loop`.
static bool whileToFor(scf::WhileOp loop) {
  Block &before = loop.getBefore().front(), &after = loop.getAfter().front();
  auto condition = cast<scf::ConditionOp>(before.getTerminator());
  auto compare = condition.getCondition().getDefiningOp<arith::CmpIOp>();
  if (!compare || compare->getBlock() != &before || compare.getPredicate() != arith::CmpIPredicate::slt)
    return false;
  auto counter = dyn_cast<BlockArgument>(compare.getLhs());
  Value bound = compare.getRhs();
  Operation *boundDef = bound.getDefiningOp();
  if (!counter || counter.getOwner() != &before ||
      (boundDef ? loop->isAncestor(boundDef) : loop->isAncestor(bound.getParentBlock()->getParentOp())))
    return false;
  const unsigned position = counter.getArgNumber();
  std::optional<unsigned> forwarded;
  for (auto [i, v] : llvm::enumerate(condition.getArgs()))
    if (v == counter)
      forwarded = i;
  if (!forwarded)
    return false;
  auto yield = cast<scf::YieldOp>(after.getTerminator());
  auto next = yield.getOperand(position).getDefiningOp<arith::AddIOp>();
  Value afterCounter = after.getArgument(*forwarded);
  if (!next || !((next.getLhs() == afterCounter && matchPattern(next.getRhs(), m_One())) ||
                 (next.getRhs() == afterCounter && matchPattern(next.getLhs(), m_One()))))
    return false;
  Location loc = loop.getLoc();
  OpBuilder builder(loop);
  Value lower = loop.getInits()[position];
  Type type = counter.getType();
  Value one = type.isIndex() ? arith::ConstantIndexOp::create(builder, loc, 1).getResult()
                             : arith::ConstantIntOp::create(builder, loc, type, 1).getResult();
  SmallVector<Value> inits;
  for (auto [i, v] : llvm::enumerate(loop.getInits()))
    if (i != position)
      inits.push_back(v);
  // The condition's region cloned with the counter at `iv` and the other
  // values at `values`.
  auto runBefore = [&](OpBuilder &at, IRMapping &mapping, Value iv, ValueRange values) {
    mapping.map(counter, iv);
    unsigned k = 0;
    for (BlockArgument argument : before.getArguments())
      if (argument != counter)
        mapping.map(argument, values[k++]);
    for (Operation &op : before.without_terminator())
      at.clone(op, mapping);
  };
  auto forOp = scf::ForOp::create(
      builder, loc, lower, bound, one, inits, [&](OpBuilder &b, Location l, Value iv, ValueRange values) {
        IRMapping mapping;
        runBefore(b, mapping, iv, values);
        for (auto [argument, v] : llvm::zip(after.getArguments(), condition.getArgs()))
          mapping.map(argument, mapping.lookupOrDefault(v));
        for (Operation &op : after.without_terminator())
          b.clone(op, mapping);
        SmallVector<Value> yielded;
        for (auto [i, v] : llvm::enumerate(yield.getOperands()))
          if (i != position)
            yielded.push_back(mapping.lookupOrDefault(v));
        scf::YieldOp::create(b, l, yielded);
      });
  builder.setInsertionPointAfter(forOp);
  IRMapping mapping;
  runBefore(builder, mapping, arith::MaxSIOp::create(builder, loc, lower, bound), forOp.getResults());
  for (auto [result, v] : llvm::zip(loop.getResults(), condition.getArgs()))
    result.replaceAllUsesWith(mapping.lookupOrDefault(v));
  loop.erase();
  return true;
}

static int64_t cleaveFoldLoopsIntoGrids(MlirOperation op) {
  auto module = dyn_cast<ModuleOp>(unwrap(op));
  if (!module)
    return 0;
  // Host `scf.while`s launching kernels made `scf.for`s first, inner ones
  // first (post-order).
  SmallVector<scf::WhileOp> whiles;
  module.walk([&](scf::WhileOp loop) {
    if (loop->getParentOfType<gpu::LaunchOp>() || loop->getParentOfType<gpu::GPUModuleOp>())
      return;
    bool launches = false;
    loop->walk([&](gpu::LaunchOp) { launches = true; });
    if (launches)
      whiles.push_back(loop);
  });
  for (scf::WhileOp loop : whiles)
    whileToFor(loop);
  SmallVector<scf::ForOp> loops; // post-order: inner loops first
  module.walk([&](scf::ForOp loop) {
    if (!loop->getParentOfType<gpu::LaunchOp>() && !loop->getParentOfType<gpu::GPUModuleOp>())
      loops.push_back(loop);
  });
  FreshMemo fresh;
  int64_t folded = 0;
  for (scf::ForOp loop : loops)
    if (foldLoopIntoGrids(loop, module, fresh)) {
      ++folded;
      fresh.clear(); // its values' addresses may be reused
    }
  return folded;
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
    for (auto [i, argument] : llvm::enumerate(operands)) {
      Value operand = argument;
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
      // A vector view of a buffer (`cleave-tile-matmuls-for-gpu`): the buffer
      // it views, bound as it is.
      if (auto view = operand.getDefiningOp<UnrealizedConversionCastOp>();
          view && view->getNumOperands() == 1 && isa<MemRefType>(view->getOperand(0).getType()))
        operand = view->getOperand(0);
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

  // The vector views the launches took, bound as their buffers: unused now.
  SmallVector<Operation *> views;
  module.walk([&](UnrealizedConversionCastOp view) {
    if (view->use_empty() && view->getNumOperands() == 1 && isa<MemRefType>(view->getOperand(0).getType()) &&
        isa<MemRefType>(view->getResult(0).getType()))
      views.push_back(view);
  });
  for (Operation *view : views)
    view->erase();
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
// Vulkan backend: products for the tensor cores become calls to the runtime's
// GEMMs before the kernels are formed, their operands bound as a kernel's are
// (`memrefSpan`, the mirrors' keys), the transpositions read from the indexing
// maps:
// - a product the program asked to compute in bfloat16 (a `linalg.matmul`
//   marked `cleave.compute = "bf16"`, on float32 memrefs: `stdlib/linalg`'s
//   `MatMulBf16`), `cleave_vk_gemm_bf16(a, a_bytes, b, b_bytes, c, c_bytes, m,
//   n, k, ta, tb)`, which rounds the operands as it loads them. A shape the
//   kernel can't take is an error, located: computed in f32 instead it would
//   compute something else.
// - a product of bf16 tensors (a `linalg.matmul` with bf16 operands into f32,
//   `mlir_lower.rs::widened_16_bit_float_tensor`),
//   `cleave_vk_gemm_bf16_operands`, the same arguments. A shape the kernel
//   can't take stays a generated kernel, which computes the same values.
// The kernels need whole row-major matrices (an identity layout) with M and N
// multiples of 128 and K of 32.
struct LowerGpuGemmsPass : PassWrapper<LowerGpuGemmsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerGpuGemmsPass)
  StringRef getArgument() const final { return "cleave-lower-gpu-gemms"; }
  StringRef getName() const final { return "cleave-lower-gpu-gemms"; }
  StringRef getDescription() const final {
    return "bf16 products to the Vulkan runtime's tensor-core GEMM";
  }
  void runOnOperation() final {
    ModuleOp module = getOperation();
    // (the product, whether its operands are bf16 in memory)
    SmallVector<std::pair<linalg::MatmulOp, bool>> products;
    auto elementOf = [](Value v) -> Type {
      auto type = dyn_cast<MemRefType>(v.getType());
      return type ? type.getElementType() : Type();
    };
    module.walk([&](linalg::MatmulOp product) {
      if (product->getParentOfType<gpu::GPUModuleOp>() || !product.hasPureBufferSemantics())
        return;
      auto compute = product->getAttrOfType<StringAttr>(kComputeAttr);
      if (compute && compute.getValue() == "bf16") {
        products.push_back({product, false});
        return;
      }
      Type a = elementOf(product.getDpsInputs()[0]), b = elementOf(product.getDpsInputs()[1]);
      Type c = elementOf(product.getDpsInits()[0]);
      if (a && b && c && a.isBF16() && b.isBF16() && c.isF32())
        products.push_back({product, true});
    });
    if (products.empty())
      return;
    MLIRContext *context = module.getContext();
    Type i64 = IntegerType::get(context, 64);
    for (StringRef name : {StringRef(kGemmBf16), StringRef(kGemmBf16Operands)}) {
      if (module.lookupSymbol<func::FuncOp>(name))
        continue;
      OpBuilder top(context);
      top.setInsertionPointToStart(module.getBody());
      auto declaration = func::FuncOp::create(top, module.getLoc(), name,
                                              FunctionType::get(context, SmallVector<Type>(11, i64), {}));
      declaration.setPrivate();
    }
    AffineExpr d0 = getAffineDimExpr(0, context), d1 = getAffineDimExpr(1, context), d2 = getAffineDimExpr(2, context);
    auto map = [&](AffineExpr x, AffineExpr y) { return AffineMap::get(3, 0, {x, y}, context); };
    bool failed = false;
    for (auto [product, bf16Operands] : products) {
      Location loc = product.getLoc();
      Value a = product.getDpsInputs()[0], b = product.getDpsInputs()[1], c = product.getDpsInits()[0];
      auto whole = [](Value v) {
        auto type = dyn_cast<MemRefType>(v.getType());
        return type && type.getRank() == 2 && type.hasStaticShape() && type.getLayout().isIdentity();
      };
      SmallVector<AffineMap> maps = product.getIndexingMapsArray();
      bool ta = maps[0] == map(d2, d0), tb = maps[1] == map(d1, d2);
      bool knownMaps = (ta || maps[0] == map(d0, d2)) && (tb || maps[1] == map(d2, d1)) && maps[2] == map(d0, d1);
      if (!whole(a) || !whole(b) || !whole(c) || !knownMaps) {
        if (bf16Operands)
          continue; // a generated kernel computes the same values
        product.emitError("a bf16 product on the GPU needs whole row-major f32 matrices "
                          "(doc/plan-quantization.md)");
        failed = true;
        continue;
      }
      auto aType = cast<MemRefType>(a.getType()), cType = cast<MemRefType>(c.getType());
      int64_t m = cType.getDimSize(0), n = cType.getDimSize(1), k = aType.getDimSize(ta ? 0 : 1);
      if (m % 128 != 0 || n % 128 != 0 || k % 32 != 0) {
        if (bf16Operands)
          continue;
        product.emitError() << "a bf16 product of " << m << "x" << n << "x" << k
                            << " on the GPU needs M and N multiples of 128 and K of 32 "
                               "(doc/plan-quantization.md)";
        failed = true;
        continue;
      }
      OpBuilder builder(product);
      auto constant = [&](int64_t v) -> Value { return arith::ConstantIntOp::create(builder, loc, i64, v); };
      auto [aPtr, aBytes] = memrefSpan(builder, loc, a, /*fromBase=*/true);
      auto [bPtr, bBytes] = memrefSpan(builder, loc, b, /*fromBase=*/true);
      auto [cPtr, cBytes] = memrefSpan(builder, loc, c, /*fromBase=*/true);
      func::CallOp::create(builder, loc, bf16Operands ? kGemmBf16Operands : kGemmBf16, TypeRange{},
                           ValueRange{aPtr, aBytes, bPtr, bBytes, cPtr, cBytes, constant(m), constant(n),
                                      constant(k), constant(ta), constant(tb)});
      product.erase();
    }
    if (failed)
      signalPassFailure();
  }
};

CLEAVE_COUNTING_PASS(SyncHostAccessesPass, "cleave-sync-host-accesses",
                     "host accesses to memory kernels may share, reported to the Vulkan runtime",
                     cleaveSyncHostAccesses)
CLEAVE_COUNTING_PASS(FoldLoopsIntoGridsPass, "cleave-fold-loops-into-grids",
                     "host loops launching kernels folded into the kernels' grids",
                     cleaveFoldLoopsIntoGrids)
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
  PassRegistration<LowerGpuGemmsPass>();
  PassRegistration<FoldLoopsIntoGridsPass>();
  PassRegistration<LowerGpuLaunchesPass>();
}
