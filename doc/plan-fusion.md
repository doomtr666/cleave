# Plan: fusing row pipelines into one kernel — attention without its score matrix

Status (2026-10-16): phases 1 to 4 done (`cleave-fuse-row-pipelines`,
`gpu.cpp`), on by default, the CPU's values to the bit in both
(`CLEAVE_GPU_FUSE=0` at compile time turns it off, to compare).

| attention forward, bf16, 12 heads | unfused | fused |
|---|---|---|
| L = 256, whole rows (phase 1, 16 rows a workgroup) | 30.5 us | 20.1 us |
| L = 1024, blocks of 64 (phase 2, 32 rows a workgroup), 80 launches | 15.6 ms | 15.1 ms |

nanoLM (`bench 0 1 3`, device time): 13.85 s -> 13.27 s, the forward
attention 1.96 -> 1.63 s (fused), the backward 2.07 -> 1.81 s (its separate
products staged, below).

What the measurements taught (L = 1024, Nsight Graphics' GPU trace and its
per-instruction export):

- 16 rows a workgroup and `Q` reloaded at every block of every pass moved
  ~365 MB through the L2 (473 us); 64 rows and `Q`'s fragments loaded once
  into registers, ~50 MB (305 us; 32 rows, 269).
- The rest was 46% of the samples on fragment loads from device memory
  (long scoreboard, LG throttle): each a scatter of small accesses. Staging
  the `b` operands' tiles in the workgroup's memory 16 bytes at a time
  (`VectorSource`, `tileVectors`, `loadFragmentFromVectors`: the runtime
  GEMM's `uvec4` tiles), loaded a block ahead into registers, took it to
  ~174 us. The same staging serves every fragment product
  (`tileProductOnFragments`): `Q Kᵀ` 29 -> 20 us, `P V` 37 -> 28 (6 heads).
- Barriers ordering the workgroup's memory only (`MEMBAR`s were 5.5%); the
  last blocks of rows first (the causal tail) was 7% slower: nearly every
  workgroup is resident at once.

Phases 3 and 4 done differently than planned: the online form and the
blocked backward are stdlib algorithms run by `cleave-block-programs`, not
backend rewrites (below).

### Phase 4: the online form, a stdlib algorithm (`cleave-block-programs`)

The online softmax rounds differently (each weight to bf16 against the
running maximum of its block of keys, not once normalized): a precision
choice, which the backend must not make silently. So it's an algorithm the
stdlib states, `nn::online_causal_attention` (blocks of 32 rows by 64 keys,
the block of keys part of its definition, `L` a multiple of 64), and the
bf16 `CausalAttention` impl's forward; `exact_causal_attention` keeps the
three-pass rounding (and `cleave-fuse-row-pipelines` still fuses it). The
gradient stays the exact attention's.

What the backend learned is general, not attention: `cleave-block-programs`
makes the innermost loop of independent blocks with fragment products in it
(each iteration writing its own block of an outer buffer) one kernel, a
workgroup per iteration, Triton-style:

- the body's allocations in the workgroup's memory, a freed one reused by a
  later allocation of the same type (46.9 -> 36.7 KB);
- products on fragments, every subgroup its share of the output's
  fragments; operands from device memory staged into two role tiles (`a`,
  `b`) shared by every product; an operand defined in the block but read in
  a loop inside it (the block of queries) its own tile, loaded once;
- `scf.parallel` row loops a group of threads a row (`emitRows`, tree
  reductions), other parallel loops and elementwise `linalg` shared among
  the threads, loops over blocks kept uniform, other writes the first
  thread's; a barrier after each step.

| L = 1024, 12 heads, 80 launches | per launch |
|---|---|
| exact, fused rows (`exact_causal_attention`) | 176.5 + 2 x 8.3 us |
| online, block program, 4 subgroups | 406 -> 240 (rows) -> 190 us (staging) |
| online, block program, 8 subgroups (`CLEAVE_GPU_BLOCK_SUBGROUPS`) | 172 us, one kernel |
| + `acc = fma(P, V, acc)`, blocks of keys and values loaded a turn ahead | 170 us |

Q in its own tile changed nothing measurable (it's 4 KB); doubling the
subgroups did (2 workgroups an SM by their memory: 8 warps -> 16), 16 was
slower (207 us). `fma` with bf16 operands into an f32 accumulator
(`stdlib/linalg`) changed nothing here (the `pv` buffer was already the
scores' one, reused), but it's what the backward's `dK`, `dV` accumulators
need; writing it took three generic fixes: a tensor's `t[i, j] = v` keeps its
index array on the stack (`stack_index_arrays`), counted `scf.while`s become
`scf.for`s before the deallocation (`upliftCountedWhiles`: a buffer updated
in place and yielded back no longer seen as of unknown origin, nor its loop
as sequential), and upstream's uplift's exit value of the counter corrected
(it gave the last turn's).

Per product, the block program is no faster than the separate staged
products (~6.5 us vs ~5.5 us a product a head at L = 1024): the forward won
by removing the softmax's kernels and their L x L traffic, not by faster
products. Each product is one 16 x 16 fragment a subgroup between two
barriers.

## Why

Attention per head is three kernels, each going through device memory
(`stdlib/nn`, `CausalAttention`): `S = Q Kᵀ` (L x L, f32: 4 MiB a head at L =
1024), the softmax reading `S` and writing `P`, `O = P V` reading `P`. The
products are cheap (2·L²·d FLOPs); writing and reading `S` and `P` is what
costs (`Q Kᵀ` runs at the memory's speed, `doc/plan-quantization.md` §4b.4).
FlashAttention (Dao et al., 2022) never materializes them. Seen from the
compiler, it is three general transformations, not an algorithm to write:

- **fusion** of producer and consumers along a parallel dimension they share
  (attention's query rows `i`: the softmax reads row `i` of `S` only, `P V`
  computes row `i` of `O` from row `i` of `P` only);
- **contraction** of the intermediates to the workgroup's memory (`S`, `P`:
  a block of rows each, never in device memory);
- when a block of rows doesn't fit, **tiling of the other dimension `j`**
  (the keys), the producer recomputed for each loop that reads it
  (rematerialization: the softmax's three passes over a row each recompute
  their block of scores), or, later, carried as an online reduction whose
  state is reassociated by a monoid law (`(m, l, o)`, the online softmax).

Purely backend: the stdlib keeps the naive definition, any program of the
same shape fuses. Fusion and contraction compute the same arithmetic as
today (only where intermediates live changes); recomputation computes the
same values again; only the online form would reorder sums (stated as such
when it comes, like split-K and the fragments).

## The pattern (`cleave-fuse-row-pipelines`, Vulkan backend, before kernels form)

In one block, after `cleave-parallelize-loops` (`CLEAVE_DUMP_GPU_LOOPS`):

- **producers**: products (`linalg` contractions, bf16 operands into f32: on
  fragments) each writing a fresh buffer `S` whose rows are the product's
  `m`;
- **a row loop**: `scf.parallel (i)`, its body scalar operations and inner
  `scf.for j` loops: reductions (carried values, no stores) and maps (no
  carried values, stores to `P[i, j]` only), every access of a producer's
  buffer at row `i`, every store at row `i`;
- **consumers**: products whose `a` is a buffer `P` the row loop writes, as
  `P[m, k]` (their output's rows are `P`'s).

All three become one `gpu.launch`, a workgroup per block of `R` rows (16, a
fragment's height), 4 subgroups:

1. **row statistics**: the reduction loops in order, each over the blocks of
   `j` its rows' bounds reach: the block of each producer it reads computed
   on fragments into workgroup memory (`R x Bj`), then the loop's
   iterations shared among the row's threads, partial values combined by a
   tree in workgroup memory (`cleave-cooperative-rows`' shape, a row per
   group of threads);
2. **maps and consumers**: per block of `j` the consumers' live slices reach
   (`zeroOn`), the producers' blocks again, every map loop's iterations in
   that block written to the block of `P` in workgroup memory, then each
   consumer's accumulators (fragments, registers) `+= P_blk · B_blk`;
3. the consumers' accumulators stored to their results.

`Bj = L` when the blocks of every producer and `P` fit the workgroup memory
(no recomputation: phase 1), else a block of 64 and every producer recomputed
per loop (phase 2).

## Phases

1. ✅ Phase 1, contraction without recomputation: the pattern, the fused
   kernel with `Bj = L`, the CPU's values on attention at L = 256, 28% faster
   (`vulkan.rs::attention_fuses_into_one_kernel_with_whole_rows`). The pieces,
   reusable: fragment products inside a kernel reading and writing
   workgroup memory (`FragmentProduct`, `loadFragment`, `storeFragment`); a
   row loop's body run by a group of threads per row.
2. ✅ Phase 2, recomputation: `Bj` = 64 < L, nanoLM's L = 1024, the CPU's
   values (`attention_fuses_into_one_kernel_in_blocks`), faster than the
   kernels it replaces once its tiles were staged (the table above).
3. ✅ Phase 3, the backward (FlashAttention-2's): the forward keeps each
   row's log-sum-exp (`online_causal_attention_stats`); the backward
   (`online_causal_attention_backward`) computes `dot = dO · O` by row, then
   a block program per block of 32 keys (`S`, `dP`, the weights from the
   statistics, `dS`, `dV += Pᵀ dO`, `dK += dSᵀ Q` accumulated with `fma`,
   `dS` kept by blocks) and one per block of 32 queries (`dQ += dS K`, the
   causal blocks only). The statistics reach the backward through the
   algebra: `CausalAttention<X, S, R>`'s differentiated method is
   `causal_attention_stats` (`(X, R)`), its rule's call the forward's own
   (one e-class), its upstream a pair read by field (`u[0]`, which adjoint
   rules now take). `dot` from the bf16 output, within its rounding of
   `rowsum(dP * P)`; the rest the exact attention's to f32 rounding
   (`vulkan.rs::bf16_attention_gradient_reads_its_forwards_statistics`).
   Its first nanoLM run went NaN from the second micro-batch on: the pair
   adjoints `(u, zero)` of every layer shared one zero (one e-class), written
   in place into the first pair's field and read by the others through it
   after that pair was released (`mlir_lower.rs::build_tensor_descriptor_
   value` now copies a value stored into several structs;
   `examples.rs::a_tensor_shared_by_two_gradient_pairs_is_never_read_freed`,
   under the pool's checks, which now also guard every block's end with a
   red zone, `CLEAVE_DEBUG_POOL`). The small nanoLM then matches the exact
   attention's loss curve (9.0254, 8.9706, 8.9027 at steps 4, 14, 20).
4. ✅ Phase 4, the online form: one pass over `j`, the row state carried and
   rescaled — as a stdlib algorithm (`online_causal_attention`) run by
   `cleave-block-programs`, not a backend rewrite: its rounding is a choice
   the program states (`vulkan.rs::online_attention_*`).

## Risks

- KHR fragments are opaque per element: every row operation goes through
  workgroup memory (VK_NV_cooperative_matrix2 has row reductions; NVIDIA
  only, later if measured worth it).
- Workgroup memory: 48 KiB (NVIDIA's Vulkan limit) bounds `R x Bj`.
- Subgroups of 32 assumed, as by the fragment products.
