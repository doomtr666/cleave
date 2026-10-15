// `C = A B + C` on one 16x16x16 fragment, A and B bfloat16, C float32 (set 0, bindings 0-2, row
// major, leading dimension 16), one subgroup of 32 invocations: a hand-written cooperative-matrix
// kernel in the `spirv` dialect, the tensor-core probe of `doc/plan-quantization.md` (phase 0).
// MLIR 22's `!gpu.mma_matrix` rejects bf16, so the `spirv` ops directly; SPIR-V 1.6 with the
// Vulkan memory model, which cooperative matrices in a shader require.
spirv.module Logical Vulkan requires #spirv.vce<v1.6, [Shader, CooperativeMatrixKHR, BFloat16TypeKHR, BFloat16CooperativeMatrixKHR, StorageBuffer16BitAccess, VulkanMemoryModel], [SPV_KHR_cooperative_matrix, SPV_KHR_bfloat16, SPV_KHR_16bit_storage, SPV_KHR_vulkan_memory_model]> {
  spirv.GlobalVariable @A bind(0, 0) : !spirv.ptr<!spirv.struct<(!spirv.rtarray<bf16, stride=2> [0]), Block>, StorageBuffer>
  spirv.GlobalVariable @B bind(0, 1) : !spirv.ptr<!spirv.struct<(!spirv.rtarray<bf16, stride=2> [0]), Block>, StorageBuffer>
  spirv.GlobalVariable @C bind(0, 2) : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0]), Block>, StorageBuffer>
  spirv.func @mma() "None" {
    %zero = spirv.Constant 0 : i32
    %stride = spirv.Constant 16 : i32
    %a = spirv.mlir.addressof @A : !spirv.ptr<!spirv.struct<(!spirv.rtarray<bf16, stride=2> [0]), Block>, StorageBuffer>
    %b = spirv.mlir.addressof @B : !spirv.ptr<!spirv.struct<(!spirv.rtarray<bf16, stride=2> [0]), Block>, StorageBuffer>
    %c = spirv.mlir.addressof @C : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0]), Block>, StorageBuffer>
    %pa = spirv.AccessChain %a[%zero, %zero] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<bf16, stride=2> [0]), Block>, StorageBuffer>, i32, i32 -> !spirv.ptr<bf16, StorageBuffer>
    %pb = spirv.AccessChain %b[%zero, %zero] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<bf16, stride=2> [0]), Block>, StorageBuffer>, i32, i32 -> !spirv.ptr<bf16, StorageBuffer>
    %pc = spirv.AccessChain %c[%zero, %zero] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0]), Block>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
    %ma = spirv.KHR.CooperativeMatrixLoad %pa, %stride, <RowMajor> : !spirv.ptr<bf16, StorageBuffer>, i32 -> !spirv.coopmatrix<16x16xbf16, Subgroup, MatrixA>
    %mb = spirv.KHR.CooperativeMatrixLoad %pb, %stride, <RowMajor> : !spirv.ptr<bf16, StorageBuffer>, i32 -> !spirv.coopmatrix<16x16xbf16, Subgroup, MatrixB>
    %mc = spirv.KHR.CooperativeMatrixLoad %pc, %stride, <RowMajor> : !spirv.ptr<f32, StorageBuffer>, i32 -> !spirv.coopmatrix<16x16xf32, Subgroup, MatrixAcc>
    %md = spirv.KHR.CooperativeMatrixMulAdd %ma, %mb, %mc : !spirv.coopmatrix<16x16xbf16, Subgroup, MatrixA>, !spirv.coopmatrix<16x16xbf16, Subgroup, MatrixB> -> !spirv.coopmatrix<16x16xf32, Subgroup, MatrixAcc>
    spirv.KHR.CooperativeMatrixStore %pc, %md, %stride, <RowMajor> : !spirv.ptr<f32, StorageBuffer>, !spirv.coopmatrix<16x16xf32, Subgroup, MatrixAcc>, i32
    spirv.Return
  }
  spirv.EntryPoint "GLCompute" @mma, @A, @B, @C
  spirv.ExecutionMode @mma "LocalSize", 32, 1, 1
}
