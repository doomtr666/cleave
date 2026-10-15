// Tensor-core throughput probe (`doc/plan-quantization.md`, phase 0): each subgroup loads one
// bfloat16 A and B fragment (16x16x16), then runs 2048 iterations of 4 independent
// `C += A B` (float32 accumulators, so no single dependency chain bounds it), and stores their sum
// so the work isn't dead. 128 invocations per workgroup, 4 subgroups. 2 * 16^3 flops per product.
spirv.module Logical Vulkan requires #spirv.vce<v1.6, [Shader, CooperativeMatrixKHR, BFloat16TypeKHR, BFloat16CooperativeMatrixKHR, StorageBuffer16BitAccess, VulkanMemoryModel], [SPV_KHR_cooperative_matrix, SPV_KHR_bfloat16, SPV_KHR_16bit_storage, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @A bind(0, 0) : !spirv.ptr<!spirv.struct<(!spirv.rtarray<bf16, stride=2> [0]), Block>, StorageBuffer>
  spirv.GlobalVariable @B bind(0, 1) : !spirv.ptr<!spirv.struct<(!spirv.rtarray<bf16, stride=2> [0]), Block>, StorageBuffer>
  spirv.GlobalVariable @C bind(0, 2) : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0]), Block>, StorageBuffer>
  spirv.func @peak() "None" {
    %zero = spirv.Constant 0 : i32
    %one = spirv.Constant 1 : i32
    %iters = spirv.Constant 2048 : i32
    %stride = spirv.Constant 16 : i32
    %a = spirv.mlir.addressof @A : !spirv.ptr<!spirv.struct<(!spirv.rtarray<bf16, stride=2> [0]), Block>, StorageBuffer>
    %b = spirv.mlir.addressof @B : !spirv.ptr<!spirv.struct<(!spirv.rtarray<bf16, stride=2> [0]), Block>, StorageBuffer>
    %c = spirv.mlir.addressof @C : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0]), Block>, StorageBuffer>
    %pa = spirv.AccessChain %a[%zero, %zero] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<bf16, stride=2> [0]), Block>, StorageBuffer>, i32, i32 -> !spirv.ptr<bf16, StorageBuffer>
    %pb = spirv.AccessChain %b[%zero, %zero] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<bf16, stride=2> [0]), Block>, StorageBuffer>, i32, i32 -> !spirv.ptr<bf16, StorageBuffer>
    %pc = spirv.AccessChain %c[%zero, %zero] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0]), Block>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
    %ma = spirv.KHR.CooperativeMatrixLoad %pa, %stride, <RowMajor> : !spirv.ptr<bf16, StorageBuffer>, i32 -> !spirv.coopmatrix<16x16xbf16, Subgroup, MatrixA>
    %mb = spirv.KHR.CooperativeMatrixLoad %pb, %stride, <RowMajor> : !spirv.ptr<bf16, StorageBuffer>, i32 -> !spirv.coopmatrix<16x16xbf16, Subgroup, MatrixB>
    %init = spirv.KHR.CooperativeMatrixLoad %pc, %stride, <RowMajor> : !spirv.ptr<f32, StorageBuffer>, i32 -> !spirv.coopmatrix<16x16xf32, Subgroup, MatrixAcc>
    %v0 = spirv.Variable : !spirv.ptr<!spirv.coopmatrix<16x16xf32, Subgroup, MatrixAcc>, Function>
    %v1 = spirv.Variable : !spirv.ptr<!spirv.coopmatrix<16x16xf32, Subgroup, MatrixAcc>, Function>
    %v2 = spirv.Variable : !spirv.ptr<!spirv.coopmatrix<16x16xf32, Subgroup, MatrixAcc>, Function>
    %v3 = spirv.Variable : !spirv.ptr<!spirv.coopmatrix<16x16xf32, Subgroup, MatrixAcc>, Function>
    %vi = spirv.Variable : !spirv.ptr<i32, Function>
    spirv.Store "Function" %v0, %init : !spirv.coopmatrix<16x16xf32, Subgroup, MatrixAcc>
    spirv.Store "Function" %v1, %init : !spirv.coopmatrix<16x16xf32, Subgroup, MatrixAcc>
    spirv.Store "Function" %v2, %init : !spirv.coopmatrix<16x16xf32, Subgroup, MatrixAcc>
    spirv.Store "Function" %v3, %init : !spirv.coopmatrix<16x16xf32, Subgroup, MatrixAcc>
    spirv.Store "Function" %vi, %zero : i32
    spirv.mlir.loop {
      spirv.Branch ^header
    ^header:
      %i = spirv.Load "Function" %vi : i32
      %more = spirv.SLessThan %i, %iters : i32
      spirv.BranchConditional %more, ^body, ^merge
    ^body:
      %x0 = spirv.Load "Function" %v0 : !spirv.coopmatrix<16x16xf32, Subgroup, MatrixAcc>
      %x1 = spirv.Load "Function" %v1 : !spirv.coopmatrix<16x16xf32, Subgroup, MatrixAcc>
      %x2 = spirv.Load "Function" %v2 : !spirv.coopmatrix<16x16xf32, Subgroup, MatrixAcc>
      %x3 = spirv.Load "Function" %v3 : !spirv.coopmatrix<16x16xf32, Subgroup, MatrixAcc>
      %y0 = spirv.KHR.CooperativeMatrixMulAdd %ma, %mb, %x0 : !spirv.coopmatrix<16x16xbf16, Subgroup, MatrixA>, !spirv.coopmatrix<16x16xbf16, Subgroup, MatrixB> -> !spirv.coopmatrix<16x16xf32, Subgroup, MatrixAcc>
      %y1 = spirv.KHR.CooperativeMatrixMulAdd %ma, %mb, %x1 : !spirv.coopmatrix<16x16xbf16, Subgroup, MatrixA>, !spirv.coopmatrix<16x16xbf16, Subgroup, MatrixB> -> !spirv.coopmatrix<16x16xf32, Subgroup, MatrixAcc>
      %y2 = spirv.KHR.CooperativeMatrixMulAdd %ma, %mb, %x2 : !spirv.coopmatrix<16x16xbf16, Subgroup, MatrixA>, !spirv.coopmatrix<16x16xbf16, Subgroup, MatrixB> -> !spirv.coopmatrix<16x16xf32, Subgroup, MatrixAcc>
      %y3 = spirv.KHR.CooperativeMatrixMulAdd %ma, %mb, %x3 : !spirv.coopmatrix<16x16xbf16, Subgroup, MatrixA>, !spirv.coopmatrix<16x16xbf16, Subgroup, MatrixB> -> !spirv.coopmatrix<16x16xf32, Subgroup, MatrixAcc>
      spirv.Store "Function" %v0, %y0 : !spirv.coopmatrix<16x16xf32, Subgroup, MatrixAcc>
      spirv.Store "Function" %v1, %y1 : !spirv.coopmatrix<16x16xf32, Subgroup, MatrixAcc>
      spirv.Store "Function" %v2, %y2 : !spirv.coopmatrix<16x16xf32, Subgroup, MatrixAcc>
      spirv.Store "Function" %v3, %y3 : !spirv.coopmatrix<16x16xf32, Subgroup, MatrixAcc>
      %next = spirv.IAdd %i, %one : i32
      spirv.Store "Function" %vi, %next : i32
      spirv.Branch ^continue
    ^continue:
      spirv.Branch ^header
    ^merge:
      spirv.mlir.merge
    }
    %r0 = spirv.Load "Function" %v0 : !spirv.coopmatrix<16x16xf32, Subgroup, MatrixAcc>
    %r1 = spirv.Load "Function" %v1 : !spirv.coopmatrix<16x16xf32, Subgroup, MatrixAcc>
    %r2 = spirv.Load "Function" %v2 : !spirv.coopmatrix<16x16xf32, Subgroup, MatrixAcc>
    %r3 = spirv.Load "Function" %v3 : !spirv.coopmatrix<16x16xf32, Subgroup, MatrixAcc>
    %s01 = spirv.FAdd %r0, %r1 : !spirv.coopmatrix<16x16xf32, Subgroup, MatrixAcc>
    %s23 = spirv.FAdd %r2, %r3 : !spirv.coopmatrix<16x16xf32, Subgroup, MatrixAcc>
    %sum = spirv.FAdd %s01, %s23 : !spirv.coopmatrix<16x16xf32, Subgroup, MatrixAcc>
    spirv.KHR.CooperativeMatrixStore %pc, %sum, %stride, <RowMajor> : !spirv.ptr<f32, StorageBuffer>, !spirv.coopmatrix<16x16xf32, Subgroup, MatrixAcc>, i32
    spirv.Return
  }
  spirv.EntryPoint "GLCompute" @peak, @A, @B, @C
  spirv.ExecutionMode @peak "LocalSize", 128, 1, 1
}
