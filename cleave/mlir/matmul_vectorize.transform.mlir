module attributes {transform.with_named_sequence} {
  // Per-op tile+vectorize action, applied independently to each matched
  // `linalg.matmul` -- see `@__transform_main`'s own comment for why a
  // failure here (a shape not evenly divisible by the tile sizes below,
  // e.g. a 10-wide output layer) must not abort the whole pipeline.
  transform.named_sequence @tile_and_vectorize(%m: !transform.any_op {transform.consumed}) {
    %inner0, %forall = transform.structured.tile_using_forall %m tile_sizes [1, 0, 0]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %inner1, %loops1 = transform.structured.tile_using_for %inner0 tile_sizes [0, 16, 0]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %tiled, %loops2 = transform.structured.tile_using_for %inner1 tile_sizes [0, 0, 16]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.structured.vectorize %tiled {create_named_contraction} : !transform.any_op
    transform.yield
  }

  transform.named_sequence @match_matmul(%m: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %m ["linalg.matmul"] : !transform.any_op
    transform.yield %m : !transform.any_op
  }

  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    // `failures(suppress)`: a shape whose `j`/`k` dims aren't evenly
    // divisible by the tile sizes above (this project's own real kernel has
    // one -- the final `128x10` output layer) makes `tile_and_vectorize`'s
    // own `vectorize` step fail for that op specifically -- confirmed
    // directly (`bench/mnist-pytorch/probe_two_shapes_*.mlir`): the failing
    // op is left exactly where tiling got it (still a real, named `linalg.
    // matmul`, just partially tiled with a dynamic remainder dim, never
    // vectorized) rather than corrupted or half-lowered. `foreach_match`
    // itself still reports a silenceable failure once any one action fails
    // (postponed until the end of its own walk, not per-op) -- `suppress`
    // here is what keeps that from aborting the rest of this pipeline
    // stage (`pipeline.rs`'s own `pass_manager.run().is_err()` check right
    // after this pass would otherwise treat it as fatal). The untouched
    // `linalg.matmul` left behind for that one shape still falls through to
    // this project's *old* `convert-linalg-to-affine-loops`/`affine-super-
    // vectorize` path later in the same pipeline, same as before this
    // transform-dialect stage existed at all.
    %new_root = transform.sequence %root : !transform.any_op -> !transform.any_op failures(suppress) {
    ^bb0(%arg0: !transform.any_op):
      %updated = transform.foreach_match in %arg0
          @match_matmul -> @tile_and_vectorize
        : (!transform.any_op) -> !transform.any_op
      transform.yield %updated : !transform.any_op
    }
    %f = transform.structured.match ops{["func.func"]} in %new_root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f {
      transform.apply_patterns.vector.lower_contraction lowering_strategy = "outerproduct"
    } : !transform.any_op
    transform.yield
  }
}
