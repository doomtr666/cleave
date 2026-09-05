module attributes {transform.with_named_sequence} {
  transform.named_sequence @match_matmul(%m: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %m ["linalg.matmul"] : !transform.any_op
    %parent = transform.get_parent_op %m : (!transform.any_op) -> !transform.any_op
    transform.match.operation_name %parent ["func.func", "scf.while"] : !transform.any_op
    transform.yield %m : !transform.any_op
  }

  // `%epi`'s own `operation_name["linalg.generic"]` check happens *inside*
  // the action, not `@match_matmul`'s own matcher -- `transform.get_
  // consumers_of_result` doesn't implement `MatchOpInterface` (confirmed
  // directly: absent from its own trait list in `TransformOps.td`, unlike
  // e.g. `get_parent_op`/`get_defining_op`, which both declare it), so it
  // can't be called from a strict matcher body at all ("expected operations
  // in the match part to implement MatchOpInterface", a real diagnostic
  // hit directly). A failure here (no consumer, more than one, or a
  // consumer that isn't a plain `linalg.generic` -- a matmul feeding
  // another matmul directly, say, the real shape several backward-pass
  // gradient chains have) happens before any mutation at all, leaving `%m`
  // completely untouched for Stage 2's own fallback below to pick up --
  // confirmed directly on an isolated probe matching this project's own
  // non-divisible output-layer shape (`128 -> 10`, the identical case the
  // *old*, matmul-only schedule already had to degrade gracefully for).
  // Found directly, on a minimal isolated probe, not assumed: inside a
  // `foreach_match`-invoked action specifically (unlike a bare top-level
  // `transform.named_sequence` body), one op's own silenceable failure does
  // *not* stop the rest of the action's own body from running -- confirmed
  // by placing a `transform.debug.emit_remark_at` right after a guaranteed-
  // to-fail `transform.match.operation_name` inside such an action: the
  // remark still fires. This is exactly what let the "`%epi` can be empty"
  // case below (an earlier, incomplete fix) still crash: a rejected `%epi`
  // (wrong op name, or the `num_associations` check failing) reported its
  // own silenceable failure but *execution kept going anyway*, straight
  // into `tile_using_forall`/`fuse_into_containing_op` on a handle that was
  // never actually validated. The fix is the nested `transform.sequence ...
  // failures(propagate)` below -- confirmed, on the identical probe, that a
  // failure *inside* a `failures(propagate)`-wrapped nested sequence *does*
  // correctly stop that sequence's own remaining ops, and correctly reports
  // as a failure of the single op (`transform.sequence`) it constitutes
  // within the action's own body -- which *does* stop the action (the
  // still-real "one op's failure stops the rest of a body" rule, just not
  // the one this action's own top-level ops were individually subject to).
  transform.named_sequence @fuse_and_vectorize(%m: !transform.any_op {transform.consumed}) {
    transform.sequence %m : !transform.any_op failures(propagate) {
    ^bb0(%arg0: !transform.any_op):
      %func = transform.get_parent_op %arg0 {op_name = "func.func"} : (!transform.any_op) -> !transform.any_op
      %epi = transform.get_consumers_of_result %arg0[0] : (!transform.any_op) -> !transform.any_op
      // `%epi` "can be empty" (`GetConsumersOfResult`'s own doc) -- a real
      // case here, not hypothetical: found directly, tracing the real
      // kernel, that the *input*-gradient matmul of the first hidden layer
      // (`net_grad`'s own backward pass differentiating all the way down to
      // `x`, the training batch itself) has its result consumed by exactly
      // one op, but not a fusable one (`bufferization.to_buffer` -- cleave
      // never backprops into the input data, so this gradient is computed
      // and simply discarded/spilled, never fed to another `linalg.generic`
      // the way every real epilogue is). `transform.match.operation_name`
      // alone doesn't reject an *empty* handle either (vacuously satisfies
      // "every associated op has one of these names" with zero ops to
      // violate it) -- `num_associations`+`match.param.cmpi` makes the
      // "exactly one real consumer" requirement explicit before `operation_
      // name` or any tiling step ever runs.
      %one = transform.param.constant 1 : i64 -> !transform.param<i64>
      %epi_count = transform.num_associations %epi : (!transform.any_op) -> !transform.param<i64>
      transform.match.param.cmpi eq %epi_count, %one : !transform.param<i64>
      transform.match.operation_name %epi ["linalg.generic"] : !transform.any_op
      %epi_tiled, %forall = transform.structured.tile_using_forall %epi tile_sizes [1, 0]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
      %fused_mm, %new_forall = transform.structured.fuse_into_containing_op %arg0 into %forall
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
      // See this sequence's own doc comment (above `@match_matmul`) --
      // erases the now-dead original matmul `fuse_into_containing_op`
      // cloned *from*, before `foreach_match`'s own walk can ever revisit
      // it.
      transform.apply_dce to %func : !transform.any_op
      // The epilogue's own `vectorize` -- no `create_named_contraction`,
      // it's genuinely elementwise (identity indexing maps, no reduction),
      // not a contraction; that attribute only means anything for `linalg.
      // matmul`/other `ContractionOpInterface` ops (`Vectorization.cpp`'s
      // own `vectorizeAsLinalgContraction`) -- runs *before* the matmul's
      // own J/K tiling+vectorize below, not after. Ordering found directly,
      // by tracing a real regression on the real kernel, not designed in up
      // front: the matmul's own `vectorize {create_named_contraction}`
      // (below) genuinely can fail (a real, expected, silenceable case --
      // this project's own non-divisible `128 -> 10` output layer, the
      // identical shape the *old*, matmul-only schedule already tolerated)
      // -- and since this whole sequence runs under `failures(propagate)`,
      // that failure aborts everything still to come in this block. With
      // the epilogue's own `vectorize` *after* the matmul's, that abort
      // used to strand the epilogue mid-transformation: already tiled by
      // `tile_using_forall` above (real `tensor.extract_slice`s, a
      // genuinely non-trivial, dynamic-offset layout once bufferized) but
      // *never* reaching its own `vectorize` call to clear that away --
      // confirmed directly, diffing this project's own real lowered IR
      // against the *old* schedule's: exactly two `linalg.generic` ops
      // gained a `strided<...>` operand that the old schedule's own
      // equivalent op never had, both epilogues of a matmul whose own J/K
      // vectorization fails this same way. `--affine-super-vectorize`
      // (`pipeline.rs`, the *old* fallback every unvectorized op still
      // eventually reaches) refuses that layout ("NYI: non-trivial layout
      // map") and silently falls back to a real, measured-slow gather/
      // scatter lowering instead (confirmed directly, disassembling the
      // real compiled object: two functions, zero `vfmadd`, 90-120
      // `vgatherqps`/`vscatterqps` each) -- a real, if non-fatal,
      // performance regression, not a correctness bug (`--affine-super-
      // vectorize`'s own failure here doesn't propagate as fatal either,
      // confirmed directly against melior's own `PassManager::run`).
      // Vectorizing the epilogue *first* means a later matmul-vectorize
      // failure only ever strands the *matmul* half tiled-but-unvectorized
      // -- the exact shape Stage 2's own standalone `@tile_and_vectorize`
      // already produces (and has always produced) for this identical
      // non-divisible case, already proven safe (zero `NYI` diagnostics,
      // this project's own original schedule, unchanged).
      transform.structured.vectorize %epi_tiled : !transform.any_op
      %inner1, %loops1 = transform.structured.tile_using_for %fused_mm tile_sizes [0, 16, 0]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
      %tiled_mm, %loops2 = transform.structured.tile_using_for %inner1 tile_sizes [0, 0, 16]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
      transform.structured.vectorize %tiled_mm {create_named_contraction} : !transform.any_op
      transform.yield
    }
    transform.yield
  }

  // Stage 2 -- the *original* per-op tile+vectorize action, applied
  // independently to each matched `linalg.matmul` still standing after
  // Stage 1 (i.e. every one Stage 1 didn't fuse -- no consumer, more than
  // one, or a non-`linalg.generic` shape) -- unchanged from before this
  // fusion work, see `@__transform_main`'s own comment for why a failure
  // here (a shape not evenly divisible by the tile sizes below) must not
  // abort the whole pipeline either.
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

  // Stage 0 -- backward-chain root-fusion: `net_grad`'s own backward pass,
  // walked backward from its own deepest point (`l1`'s input-gradient
  // matmul, `dx = dz1 @ W1^T` -- a genuine dead end, its result consumed
  // by nothing but a discard/materialize op, never fed to another `linalg`
  // op) all the way up to `l4`'s own bias-add/loss-gradient epilogue --
  // eight real ops (`dx`, four elementwise `relu`-derivative/bias-grad-
  // input steps, three intervening input-gradient matmuls) -- fused into
  // *one* `scf.forall`, confirmed directly on this project's own real
  // kernel, not a synthetic probe.
  //
  // Found by testing, not designed in up front, after two real surprises:
  //
  // 1. `transform.structured.fuse` (the greedy, single-call, walk-every-
  //    producer convenience op) stops dead at the *first* producer with
  //    more than one real consumer -- confirmed directly: every one of
  //    this chain's own elementwise steps (`dz1`..`dz4`) feeds *three*
  //    real consumers (the next layer's own input-gradient matmul *and*
  //    that layer's own weight-gradient matmul *and* bias-gradient
  //    reduction), never just one, so the greedy walk never gets past the
  //    very first hop. The lower-level `transform.structured.fuse_into_
  //    containing_op`, called manually, one producer at a time, does *not*
  //    share this limitation -- confirmed directly, on an isolated probe
  //    matching this exact multi-consumer shape: it makes the containing
  //    `scf.forall` produce the shared value as a genuine *extra* result
  //    (an additional `shared_outs`/`tensor.parallel_insert_slice` pair),
  //    computed once, not recomputed once per consumer -- so this stage
  //    walks the chain by hand, one `fuse_into_containing_op` call per
  //    real SSA edge, using `transform.get_producer_of_operand` (not a
  //    shape-based match -- several unrelated values in this exact chain
  //    share the same tensor shape, e.g. `dz1` and `l2->l1`'s own input-
  //    gradient matmul result, `filter_result_type` alone can't tell them
  //    apart) to name each op precisely as the real dependency graph
  //    actually connects them.
  // 2. Vectorizing all four fused matmuls (`dx`, and the `l2->l1`/`l3->l2`/
  //    `l4->l3` input-gradient matmuls) with *one* combined `transform.
  //    structured.vectorize {create_named_contraction}` call vectorized
  //    *zero* of them -- confirmed directly, not assumed: `l4->l3`'s own
  //    gradient has a real, expected, non-divisible `K=10` dimension (the
  //    same shape this project's own `128 -> 10` output layer already
  //    needs to tolerate elsewhere), and batching it together with three
  //    perfectly divisible matmuls in one `vectorize` call let its own
  //    failure silently abort the *whole* call, skipping the three that
  //    would have succeeded on their own too. Fixed by giving each matmul
  //    its own individually `failures(suppress)`-wrapped tile+vectorize
  //    sequence -- three vectorize cleanly to `vector.contract`, the
  //    fourth is left exactly where tiling got it, same as every other
  //    non-divisible matmul this project's own schedule already tolerates.
  //
  // Stops deliberately at `l4`'s own bias-add/loss-gradient epilogue --
  // *not* pulled further back into `l4`'s own forward matmul (`128 -> 10`,
  // already the schedule's own known non-divisible case) or the forward
  // pass generally, which Stage 1 below already fuses on its own terms;
  // crossing that boundary here would mean this stage and Stage 1 fighting
  // over the same ops. Weight-/bias-gradient matmuls/reductions
  // deliberately stay untouched, standalone, Stage 2's own fallback,
  // unchanged -- they consume this stage's own new, real, whole-tensor
  // `shared_outs` results (`dz1`..`dz4`) from *outside* this region, no
  // redundant recomputation, exactly the same non-redundant sharing
  // `fuse_into_containing_op` itself already guarantees for every other
  // consumer of these values.
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    %after_root_fuse = transform.sequence %root : !transform.any_op -> !transform.any_op failures(suppress) {
    ^bb0(%arg0: !transform.any_op):
      // The whole real body lives inside a *nested* `failures(propagate)`
      // sequence, not directly in this outer `failures(suppress)` one --
      // found directly, not assumed, on an isolated probe: a plain op's
      // own silenceable failure (`transform.match.param.cmpi` specifically,
      // confirmed on this exact op) does *not* stop subsequent plain ops in
      // the *same* `failures(suppress)` body the way it does in a nested
      // `failures(propagate)` one (the identical asymmetry `@fuse_and_
      // vectorize`'s own doc comment already documents for `foreach_match`
      // actions specifically -- this is a second, distinct instance of the
      // same real sharp edge, not inside `foreach_match` this time at all).
      // Without this nesting, the precondition check below (needed because
      // `filter_result_type` is only ever hit for this project's own real
      // `B=32` kernel -- a genuinely empty handle for any other shape would
      // otherwise carry all the way through to `fuse_into_containing_op`'s
      // own hard, non-silenceable "got 0" error) silently gets ignored and
      // every op after it still runs on invalidated handles.
      transform.sequence %arg0 : !transform.any_op failures(propagate) {
      ^bb1(%arg1: !transform.any_op):
      %dx = transform.structured.match ops{["linalg.matmul"]}
        filter_result_type = tensor<32x784xf32> in %arg1 : (!transform.any_op) -> !transform.any_op
      %dx_one = transform.param.constant 1 : i64 -> !transform.param<i64>
      %dx_count = transform.num_associations %dx : (!transform.any_op) -> !transform.param<i64>
      transform.match.param.cmpi eq %dx_count, %dx_one : !transform.param<i64>
      %dz1 = transform.get_producer_of_operand %dx[0] : (!transform.any_op) -> !transform.any_op
      %dh1 = transform.get_producer_of_operand %dz1[0] : (!transform.any_op) -> !transform.any_op
      %dz2 = transform.get_producer_of_operand %dh1[0] : (!transform.any_op) -> !transform.any_op
      %dh2 = transform.get_producer_of_operand %dz2[0] : (!transform.any_op) -> !transform.any_op
      %dz3 = transform.get_producer_of_operand %dh2[0] : (!transform.any_op) -> !transform.any_op
      %dh3 = transform.get_producer_of_operand %dz3[0] : (!transform.any_op) -> !transform.any_op
      %dz4 = transform.get_producer_of_operand %dh3[0] : (!transform.any_op) -> !transform.any_op

      %tiled0, %forall0 = transform.structured.tile_using_forall %dx tile_sizes [1, 0, 0]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
      %f1, %forall1 = transform.structured.fuse_into_containing_op %dz1 into %forall0
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
      %f2, %forall2 = transform.structured.fuse_into_containing_op %dh1 into %forall1
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
      %f3, %forall3 = transform.structured.fuse_into_containing_op %dz2 into %forall2
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
      %f4, %forall4 = transform.structured.fuse_into_containing_op %dh2 into %forall3
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
      %f5, %forall5 = transform.structured.fuse_into_containing_op %dz3 into %forall4
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
      %f6, %forall6 = transform.structured.fuse_into_containing_op %dh3 into %forall5
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
      %f7, %forall7 = transform.structured.fuse_into_containing_op %dz4 into %forall6
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

      %func0 = transform.get_parent_op %forall7 {op_name = "func.func"} : (!transform.any_op) -> !transform.any_op
      transform.apply_dce to %func0 : !transform.any_op

      // Elementwise first -- `dz1`..`dz4` are always divisible, batch-row-
      // shaped, no known failure case, but this ordering is load-bearing
      // in general (see this sequence's own doc comment, point 2's own
      // sibling finding in `@fuse_and_vectorize` below).
      transform.structured.vectorize %f1 : !transform.any_op
      transform.structured.vectorize %f3 : !transform.any_op
      transform.structured.vectorize %f5 : !transform.any_op
      transform.structured.vectorize %f7 : !transform.any_op

      // Matmuls one at a time -- see this sequence's own doc comment,
      // point 2, for why batching them together silently vectorized none
      // of them instead of three of four.
      transform.sequence %tiled0 : !transform.any_op failures(suppress) {
      ^bb0(%mm: !transform.any_op):
        %inner1, %k1 = transform.structured.tile_using_for %mm tile_sizes [0, 16, 0]
          : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
        %inner2, %k2 = transform.structured.tile_using_for %inner1 tile_sizes [0, 0, 16]
          : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
        transform.structured.vectorize %inner2 {create_named_contraction} : !transform.any_op
        transform.yield
      }
      transform.sequence %f2 : !transform.any_op failures(suppress) {
      ^bb0(%mm: !transform.any_op):
        %inner1, %k1 = transform.structured.tile_using_for %mm tile_sizes [0, 16, 0]
          : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
        %inner2, %k2 = transform.structured.tile_using_for %inner1 tile_sizes [0, 0, 16]
          : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
        transform.structured.vectorize %inner2 {create_named_contraction} : !transform.any_op
        transform.yield
      }
      transform.sequence %f4 : !transform.any_op failures(suppress) {
      ^bb0(%mm: !transform.any_op):
        %inner1, %k1 = transform.structured.tile_using_for %mm tile_sizes [0, 16, 0]
          : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
        %inner2, %k2 = transform.structured.tile_using_for %inner1 tile_sizes [0, 0, 16]
          : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
        transform.structured.vectorize %inner2 {create_named_contraction} : !transform.any_op
        transform.yield
      }
      transform.sequence %f6 : !transform.any_op failures(suppress) {
      ^bb0(%mm: !transform.any_op):
        %inner1, %k1 = transform.structured.tile_using_for %mm tile_sizes [0, 16, 0]
          : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
        %inner2, %k2 = transform.structured.tile_using_for %inner1 tile_sizes [0, 0, 16]
          : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
        transform.structured.vectorize %inner2 {create_named_contraction} : !transform.any_op
        transform.yield
      }
      transform.yield
      }
      transform.yield %arg0 : !transform.any_op
    }

    // `failures(suppress)`: a shape whose `j`/`k` dims aren't evenly
    // divisible by the tile sizes above (this project's own real kernel has
    // one -- the final `128x10` output layer) makes either action's own
    // `vectorize` step fail for that op specifically -- confirmed directly
    // (`bench/mnist-pytorch/probe_two_shapes_*.mlir`, and this fusion
    // work's own isolated probes): the failing op is left exactly where
    // tiling got it (still a real, named `linalg.matmul`, just partially
    // tiled with a dynamic remainder dim, never vectorized) rather than
    // corrupted or half-lowered. `foreach_match` itself still reports a
    // silenceable failure once any one action fails (postponed until the
    // end of its own walk, not per-op) -- `suppress` here is what keeps
    // that from aborting the rest of this pipeline stage (`pipeline.rs`'s
    // own `pass_manager.run().is_err()` check right after this pass would
    // otherwise treat it as fatal). Two separate `foreach_match` calls, not
    // one: Stage 1 fusing a matmul removes it (or, on its own internal
    // failure, leaves it untouched) before Stage 2's own fresh match ever
    // runs, so Stage 2 naturally only ever sees what Stage 1 didn't handle
    // -- no bespoke "did Stage 1 already claim this op" bookkeeping needed.
    %after_fuse = transform.sequence %after_root_fuse : !transform.any_op -> !transform.any_op failures(suppress) {
    ^bb0(%arg0: !transform.any_op):
      %updated = transform.foreach_match in %arg0
          @match_matmul -> @fuse_and_vectorize
        : (!transform.any_op) -> !transform.any_op
      transform.yield %updated : !transform.any_op
    }
    %new_root = transform.sequence %after_fuse : !transform.any_op -> !transform.any_op failures(suppress) {
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
