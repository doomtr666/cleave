"""Twin of `train_gpt` in `examples/nanolm/src/kernel.cleave` (nanoLM v2): the same transformer (a
token embedding of width `D`, `LAYERS` pre-RMSNorm blocks with rotary causal attention over heads
`DH` wide and a SwiGLU MLP `D -> HIDDEN -> D`, a final RMSNorm, the embedding reused as the output
head), from the same initial weights (`gpt_init.ckpt`, written by cleave's `bench` mode in its own
directory, apart from the real run's checkpoints), on the same batches, with the same summed loss,
gradient clipping, optimizers (Muon for the dense matrices, AdamW for the rest) and learning-rate
schedule. The sizes must match the kernel's `define`s. The losses must match cleave's.

    cargo run --release -p nanolm -- bench 0 <rounds> <steps_per_round>   (writes gpt_init.ckpt)
    poetry run python gpt.py 0 <rounds> <steps_per_round>

The same arguments as cleave's `bench` mode; the first step must be 0, the twin doesn't resume.
cleave also saves a checkpoint at the end of each round, counted in its round's time; the twin
doesn't.

`NANOLM_DEVICE=cuda` runs it on the GPU, against cleave's Vulkan build (`--features vulkan`). That
needs a CUDA build of PyTorch (`pyproject.toml`). Its precision, `NANOLM_PRECISION`:

- `f32` (the default): f32 throughout, TF32 products turned off (`allow_tf32`), the attention
  PyTorch's fused f32 kernel. The losses match an all-f32 cleave kernel's.
- `bf16`: the forward under `torch.autocast` in bfloat16, PyTorch's usual mixed precision, as the
  kernel chooses its own (`dense_forward_bf16`, `swiglu_forward_bf16`, the bf16 attention,
  `embedding_logits_bf16`): the products in bf16 with f32 accumulation, the attention PyTorch's
  fused bf16 kernel (FlashAttention), the norms, residuals and loss in f32 (the f32 biases promote
  the products' results back). A bf16 product's result is rounded to bf16 here, kept in f32 by
  cleave's: the losses are close, not equal. The speed comparison against today's kernel.

`NANOLM_COMPILE=1` compiles the loss (`torch.compile`): PyTorch's best case, its kernels fused.
"""

import math
import os
import sys
import time

import torch
import torch.nn.functional as F

from data import CACHE, DEFINES, TRAIN_SEED, VAL_SEED, VOCAB, Corpus, B, T, batch, read_checkpoint

arg = lambda i, default: int(sys.argv[i]) if len(sys.argv) > i else default
FIRST, ROUNDS, PER_ROUND, END = arg(1, 0), arg(2, 10), arg(3, 100), arg(4, 0)
if FIRST != 0:
    sys.exit("the twin starts from gpt_init.ckpt: the first step must be 0 (it doesn't resume)")
DEVICE = torch.device(os.environ.get("NANOLM_DEVICE", "cpu"))
if DEVICE.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
PRECISION = os.environ.get("NANOLM_PRECISION", "f32")
if PRECISION not in ("f32", "bf16"):
    sys.exit(f"NANOLM_PRECISION={PRECISION}: f32 or bf16")
COMPILE = os.environ.get("NANOLM_COMPILE") == "1"
print(f"device {DEVICE}, precision {PRECISION}{', compiled' if COMPILE else ''}")
D, DH, HIDDEN, LAYERS = (DEFINES[k] for k in ("WIDTH", "HEAD", "HIDDEN", "LAYERS"))
LR, MUON_LR = 0.001, DEFINES["MUON_LR"]
WARMUP, REWARM_AT, COOLDOWN = DEFINES["WARMUP"], DEFINES["REWARM_AT"], DEFINES["COOLDOWN"]
ACCUM, VAL_BATCHES = DEFINES["ACCUM"], DEFINES["VAL_BATCHES"]
CLIP = DEFINES["CLIP_PER_ROW"] * ACCUM * B * T  # the loss is summed over the step's rows
EPS = 1e-5
# cleave's `bench` directory: `-bench-vulkan` for its Vulkan build (the GPU comparison), `-bench` for
# its CPU one; the one this device compares against first, else the other (the same initial weights).
_BENCH = CACHE.parent / f"nanolm2-d{D}-l{LAYERS}-t{T}-v{VOCAB}-bench"
_CANDIDATES = [_BENCH.with_name(_BENCH.name + "-vulkan"), _BENCH] if DEVICE.type == "cuda" else [_BENCH, _BENCH.with_name(_BENCH.name + "-vulkan")]
MODEL_DIR = next((d for d in _CANDIDATES if (d / "gpt_init.ckpt").exists()), _CANDIDATES[0])
H = D // DH

corpus = Corpus()
leaves = [torch.tensor(a, device=DEVICE, requires_grad=True) for a in read_checkpoint(MODEL_DIR / "gpt_init.ckpt")]
assert len(leaves) == 1 + 16 * LAYERS + 1, f"{len(leaves)} leaves: not the v2 model's checkpoint"
it = iter(leaves)
take = lambda n: [next(it) for _ in range(n)]
(tok,) = take(1)
# Per block, in `Block`'s field order: n1; wq, wk, wv, wo (w, b each); n2; the SwiGLU's gate, up,
# down (w, b each).
blocks = [take(16) for _ in range(LAYERS)]
(nf,) = take(1)
params = leaves


def rms_norm(x, g):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * g


# RoPE, rotate-half convention: within each head, element j pairs with j + DH/2, rotated by
# pos * 10000^(-2j/DH); the position of a row is its index within its window of T.
_j = torch.arange(DH // 2, dtype=torch.float32)
_theta = (torch.arange(B * T) % T).float()[:, None] * torch.exp(-2.0 * _j / DH * math.log(10000.0))[None, :]
COS, SIN = torch.cos(_theta).to(DEVICE), torch.sin(_theta).to(DEVICE)


def rope(x):
    x = x.reshape(B * T, H, DH)
    a, b = x[..., : DH // 2], x[..., DH // 2 :]
    c, s = COS[:, None, :], SIN[:, None, :]
    return torch.cat([a * c - b * s, b * c + a * s], -1).reshape(B * T, D)


def attention(q, k, v):
    split = lambda m: m.reshape(B, T, H, DH).permute(0, 2, 1, 3)
    o = F.scaled_dot_product_attention(split(q), split(k), split(v), is_causal=True)
    return o.permute(0, 2, 1, 3).reshape(B * T, D)


def block(x, p):
    n1, wq, bq, wk, bk, wv, bv, wo, bo, n2, wg, bg, wu, bu, wd, bd = p
    h = rms_norm(x, n1)
    x = x + attention(rope(h @ wq + bq), rope(h @ wk + bk), h @ wv + bv) @ wo + bo
    h = rms_norm(x, n2)
    return x + (F.silu(h @ wg + bg) * (h @ wu + bu)) @ wd + bd


def forward_loss(x, y):
    h = tok[x.reshape(-1)]
    for p in blocks:
        h = block(h, p)
    return F.cross_entropy(rms_norm(h, nf) @ tok.T, y.reshape(-1), reduction="sum")


forward = torch.compile(forward_loss) if COMPILE else forward_loss


def loss(x, y):
    if PRECISION == "bf16":
        with torch.autocast(DEVICE.type, dtype=torch.bfloat16):
            return forward(x, y).float()
    return forward(x, y)


def on_device(tokens, seed, i):
    """Batch `i` (`data.batch`), on the device."""
    return (torch.from_numpy(a).to(DEVICE) for a in batch(tokens, seed, i))


def mean_loss(tokens, seed, first, batches):
    with torch.no_grad():
        total = sum(loss(*on_device(tokens, seed, s)).item() for s in range(first, first + batches))
    return total / (batches * B * T)


def validation(batches):
    return mean_loss(corpus.val, VAL_SEED, 0, batches)


def learning_rate(s, base, end=END):
    since = s - REWARM_AT if REWARM_AT > 0 and s >= REWARM_AT else s
    warm = base * (since + 1) / WARMUP if since < WARMUP else base
    if end > 0 and s > end - COOLDOWN:
        t = (s - (end - COOLDOWN)) / COOLDOWN
        return warm * (1.0 - 0.9 * t) if t < 1.0 else warm * 0.1
    return warm


# Muon for the dense layers' weight matrices (every `w` of `wq`/`wk`/`wv`/`wo` and the SwiGLU),
# AdamW for the rest: the embedding table (`Optimizer<Muon, Embedding>` routes it to AdamW), the
# biases and the norms' gains (a single row, `R == 1`).
muon_params = [p for blk in blocks for i, p in enumerate(blk) if i in (1, 3, 5, 7, 10, 12, 14)]
adamw_params = [p for p in params if all(p is not q for q in muon_params)]
adamw = torch.optim.AdamW(adamw_params, lr=LR, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
momentum = {id(p): torch.zeros_like(p) for p in muon_params}


def newton_schulz5(g):
    """`stdlib/optim/optim.cleave`'s `newton_schulz5`: five quintic iterations, on the wide side."""
    if g.shape[0] > g.shape[1]:
        return newton_schulz5(g.T).T
    x = g / (g.norm() + 1e-7)
    for _ in range(5):
        a = x @ x.T
        b = -4.7750 * a + 2.0315 * (a @ a)
        x = 3.4445 * x + b @ x
    return x


@torch.no_grad()
def muon_step(lr):
    """`Optimizer<Muon, Tensor<f32, R, C>>`: momentum 0.95 in its interpolation form, Nesterov, the
    orthogonalized update scaled by `sqrt(R / C)` when taller than wide, no weight decay."""
    for p in muon_params:
        buf = momentum[id(p)]
        buf.mul_(0.95).add_(p.grad, alpha=0.05)
        nesterov = 0.05 * p.grad + 0.95 * buf
        r, c = p.shape
        aspect = math.sqrt(r / c) if r > c else 1.0
        p.sub_(newton_schulz5(nesterov), alpha=lr * aspect)


# The same lines as the cleave host's `round_done` (`examples/nanolm/src/main.rs`), timed the same
# way: from one report to the next, the round's training steps plus the losses reported at its end
# (whose `item()` waits for the device).
# As in cleave, the total (`elapsed`) counts from before the first report, the rounds' minutes
# from right after it.
begin = time.perf_counter()
print(f"step 0: validation {corpus.per_token(validation(VAL_BATCHES))}, learning rate {learning_rate(0, LR):.6f}")
start = last = time.perf_counter()
for r in range(ROUNDS):
    for k in range(PER_ROUND):
        s = r * PER_ROUND + k
        for p in params:
            p.grad = None
        # `step_grad`: the step's `ACCUM` batches, their gradients summed.
        for a in range(ACCUM):
            loss(*on_device(corpus.train, TRAIN_SEED, s * ACCUM + a)).backward()
        torch.nn.utils.clip_grad_norm_(params, CLIP)
        for g in adamw.param_groups:
            g["lr"] = learning_rate(s, LR)
        adamw.step()
        muon_step(learning_rate(s, MUON_LR))
    step = (r + 1) * PER_ROUND
    # The next step's first four training batches, not trained on yet, as `gpt_train_loss` does.
    train = mean_loss(corpus.train, TRAIN_SEED, step * ACCUM, 4)
    val = validation(VAL_BATCHES)
    now = time.perf_counter()
    print(
        f"step {step}: train {train:.4f}, validation {corpus.per_token(val)}, learning rate {learning_rate(step, LR):.6f}, "
        f"{(now - last) * 1000 / PER_ROUND:.0f} ms/step, {(now - start) / 60:.1f} min elapsed",
        flush=True,
    )
    last = now
nats = validation(VAL_BATCHES)
print(f"transformer: {corpus.per_token(nats)}")
print(f"elapsed: {time.perf_counter() - begin:.2f}s")
