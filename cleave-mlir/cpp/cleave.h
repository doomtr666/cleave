// What shim.cpp and gpu.cpp share: the MLIR and LLVM headers, the
// helpers both use, and the macros defining cleave's passes.
#pragma once

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

// The base a memref is a view of (`shim.cpp`).
Value viewRoot(Value v);

// Registers the Vulkan backend's passes (`gpu.cpp`).
void registerGpuPasses();

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
