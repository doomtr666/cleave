# nanoLM: what a small transformer needs, and what cleave has

Status: inventory (2026-10-02), nothing built yet. Target of `doc/toward_first_release.md`'s nanoLM
showcase; the ML roadmap entry in `doc/backlog.md` asked for this inventory before writing the model.

## The model

A nanoGPT-style character model, trained from scratch on CPU, with a line-for-line PyTorch twin
(`bench/nanolm-pytorch`, like `bench/mnist-pytorch`).

| | value | why |
|---|---|---|
| corpus | French public-domain literature, auto-downloaded and cached like MNIST | publishable; `memory`: French on purpose |
| vocabulary | characters, ~100 (accents, « », dialogue dashes) | no tokenizer to build for v0.1 |
| context `T` | 128 | |
| width `d` | 128, `h` = 4 heads of `dh` = 32 | |
| layers `L` | 4, pre-LayerNorm, MLP width 4`d` | |
| batch `B` | 32, so `B*T` = 4096 rows per step | |
| parameters | ~0.8 M | ~20 GFLOP per step, minutes to tens of minutes on CPU |

One block, in the shape every op below is listed against (activations kept as `[B*T, d]` matrices,
batch and position flattened into rows):

```
h  = x + attention(layer_norm(x))          // causal, h heads
x' = h + mlp(layer_norm(h))                 // dense -> gelu/relu -> dense
```

and around the blocks: token and position embeddings in, a final LayerNorm, a dense head to `V`
logits, cross-entropy against the next character; generation samples from the last position's
softmax.

## Op by op

✅ exists and is tested · 🔶 exists, needs extending · ❌ to build

| piece | status | what is there / what is missing |
|---|---|---|
| dense layers, residual adds | ✅ | `Dense`, `dense_forward`, `Ring<Tensor>` add; 2D matmul with its adjoint |
| ReLU | ✅ | `Activation`, with adjoint |
| GELU | 🔶 | needs `tanh` (or `erf`) on tensors: `Transcendental<Tensor>` only has `exp` |
| cross-entropy | 🔶 | `CrossEntropy` on `[B, C]` with dense one-hot targets; for 4096 rows of ~100 classes, integer targets (no one-hot of 1.6 MB per step) are better: a sparse variant |
| Adam | ✅ | `AdamState`, generic `Trainable` optimizer; untested at this parameter count |
| token embedding | ❌ | gather rows of a `[V, d]` table by ids; adjoint: scatter-add into the table. Ids are integers, and `Tensor<T: Float>` can't hold them: ids as `[i32; N]` (arrays now come from the host directly, `extern fn ... -> [i32; N]`) |
| position embedding | ❌ (free once gather exists) | gather a `[T, d]` table with ids `0..T` repeated `B` times |
| LayerNorm | ❌ | row-wise mean/variance, gain and bias; as an algebra with a declared adjoint (the standard backward), not derived through reductions |
| attention | ❌ | per (batch, head): `softmax(mask + Q Kᵀ / √dh) V`. Needs slices of `[B*T, d]` (rows of one batch element, columns of one head), causal mask, row softmax, and the backward. See "Attention" below |
| row softmax | 🔶 | exists inside `CrossEntropy` only; a standalone `Softmax` with adjoint `y ⊙ (u − rowsum(u ⊙ y))` |
| sampling | ❌ (small) | softmax with temperature on one row, then a categorical draw (`rand`) |
| batch loading | ✅ | `extern fn batch(i: i32) -> [i32; N]` (2026-10-02) |
| checkpoints (save/restore) | ❌, first | resumable runs; fine-tuning later (phase 2). Step 0 below |

## Attention

Two ways to get there.

1. **General tensor machinery**: rank-3/4 tensors, `reshape` (a view: `[B*T, d]` ↔ `[B, T, h, dh]`),
   permutations (`[B, T, h, dh]` → `[B, h, T, dh]`), batched matmul (`linalg.batch_matmul`), each with
   its adjoint. The long-term shape of `linalg`, and what a real transformer library wants; a large
   first step (every op new, every adjoint new, rank-generic e-graph rules).
2. **Attention as one algebra with a declared adjoint**, the way `CrossEntropy` was done: `causal_
   attention(q, k, v)` on `[B*T, h*dh]` matrices, `B`, `T`, `h` compile-time. Implemented inside with
   loops over (batch, head), 2D slices (`tensor.extract_slice`/`insert_slice`), 2D matmuls and a row
   softmax; the backward is the standard formula (`dV = Pᵀ dO`, `dP = dO Vᵀ`, `dS = P ⊙ (dP −
   rowsum(dO ⊙ O))`, `dQ = dS K / √dh`, `dK = dSᵀ Q / √dh`), written the same way. Needs only a 2D
   `Slice` (with its adjoint, insertion into zeros) beyond what exists.

Proposal: **2 for v0.1**. It keeps every new piece 2D, reuses the matmul schedule that already performs,
and follows the pattern that worked for cross-entropy (the e-graph never differentiates through a loop,
it applies a declared rule). 1 stays the direction for `linalg` afterwards; 2's internals can move onto
it later without changing the model.

## Risks

- **Declared adjoints are only as right as their formulas.** Each new one gets a gradient check
  against PyTorch on small fixed inputs (values dumped by the twin), not just "training converges".
- **Cross-algebra rules**: an adjoint referencing methods of other algebras at different types hits
  the e-graph's single-agreed-type limitation (`nn.cleave`'s `Sum` comments); keep each rule's helpers
  in the same algebra, as `CrossEntropy` does.
- **Scalar loops**: anything elementwise written as loops over 4096 × 128 must not dominate a step;
  profile early (the cross-entropy episode: per-element index arrays and per-row regions, both fixed).
- **Matmul shapes**: row counts must be multiples of 8 (`doc/backlog.md`); `T`, `dh`, `B*T` all are.
- **A gradient leaving an `if` crashes** (`doc/backlog.md`): keep the model code free of it.
- **OpenMP**: the automatic parallelization is untested on this shape; measured, not assumed.

## Order of work

Each step ends with something that runs and is checked, against the PyTorch twin where there is one.

0. **Checkpoints** (first: a run that can be stopped and resumed is worth more than any single
   feature). `save(path, value)` / `restore(path, like)` over a `Save<T>` algebra: scalars, arrays,
   tensors, tuples (one pack impl), every `Trainable` model (one impl over its fields, like the
   optimizer), `AdamState`. File: magic and version, then per leaf its element type, rank, dims and
   data; shapes checked on restore (a clear error naming both, never misaligned weights); written to
   a temporary file then renamed (a run killed mid-save keeps its last good checkpoint). The random
   generator's state is saved too. Check: 10 steps straight and 5 steps, save, restore, 5 steps give
   bit-identical weights (MNIST-sized model); a shape mismatch is reported.
1. **Corpus**: French public-domain literature (a Pleias French-PD-Books shard or a Project Gutenberg
   selection, chosen on samples), downloaded and cached by the host like MNIST; normalization to a
   ~100-character vocabulary; train/validation split by books; `extern fn batch(i) -> [i32; N]`; the
   twin reads the same files.
2. **Embeddings and a first model**: token/position gather with scatter-add adjoint, cross-entropy
   on integer targets, tensor `tanh` and GELU. A bigram-and-MLP model trains end to end, its loss
   matching the twin's.
3. **LayerNorm, row softmax, 2D slices**, each gradient-checked against PyTorch on small fixed inputs.
4. **Causal attention** as one algebra with a declared adjoint (above), gradient-checked; the full
   model trains, with periodic checkpoints.
5. **Generation and performance**: sampling with temperature; time per step against the twin, single
   thread then OpenMP (minimum of a few runs, per-step times in steady state).
