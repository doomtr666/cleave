// `c[i] = a[i] + b[i]` over f32 storage buffers (set 0, bindings 0-2), 64
// invocations per workgroup: a hand-written compute kernel in the `spirv`
// dialect, for the SPIR-V serializer and the Vulkan runtime tests.
spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]> {
  spirv.GlobalVariable @a bind(0, 0) : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0]), Block>, StorageBuffer>
  spirv.GlobalVariable @b bind(0, 1) : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0]), Block>, StorageBuffer>
  spirv.GlobalVariable @c bind(0, 2) : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0]), Block>, StorageBuffer>
  spirv.GlobalVariable @gid built_in("GlobalInvocationId") : !spirv.ptr<vector<3xi32>, Input>
  spirv.func @add() "None" {
    %gidp = spirv.mlir.addressof @gid : !spirv.ptr<vector<3xi32>, Input>
    %g = spirv.Load "Input" %gidp : vector<3xi32>
    %x = spirv.CompositeExtract %g[0 : i32] : vector<3xi32>
    %zero = spirv.Constant 0 : i32
    %ap = spirv.mlir.addressof @a : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0]), Block>, StorageBuffer>
    %pa = spirv.AccessChain %ap[%zero, %x] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0]), Block>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
    %va = spirv.Load "StorageBuffer" %pa : f32
    %bp = spirv.mlir.addressof @b : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0]), Block>, StorageBuffer>
    %pb = spirv.AccessChain %bp[%zero, %x] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0]), Block>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
    %vb = spirv.Load "StorageBuffer" %pb : f32
    %s = spirv.FAdd %va, %vb : f32
    %cp = spirv.mlir.addressof @c : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0]), Block>, StorageBuffer>
    %pc = spirv.AccessChain %cp[%zero, %x] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0]), Block>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
    spirv.Store "StorageBuffer" %pc, %s : f32
    spirv.Return
  }
  spirv.EntryPoint "GLCompute" @add, @gid
  spirv.ExecutionMode @add "LocalSize", 64, 1, 1
}
