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
"""

import math
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from data import CACHE, TRAIN_SEED, VAL_SEED, VOCAB, Corpus, batch, read_checkpoint

arg = lambda i, default: int(sys.argv[i]) if len(sys.argv) > i else default
FIRST, ROUNDS, PER_ROUND = arg(1, 0), arg(2, 10), arg(3, 100)
if FIRST != 0:
    sys.exit("the twin starts from gpt_init.ckpt: the first step must be 0 (it doesn't resume)")
# The sizes are the kernel's `define`s, read from its source.
KERNEL = Path(__file__).resolve().parents[2] / "examples" / "nanolm" / "src" / "kernel.cleave"
DEFINES = {m[1]: int(m[2]) for m in re.finditer(r"^define (\w+): i32 = (\d+);", KERNEL.read_text(), re.M)}
T, D, DH, HIDDEN, LAYERS = (DEFINES[k] for k in ("CONTEXT", "WIDTH", "HEAD", "HIDDEN", "LAYERS"))
LR, MUON_LR, B = 0.001, 0.02, 32
WARMUP, DECAY_STEPS = DEFINES["WARMUP"], DEFINES["DECAY_STEPS"]
CLIP = 1.0 * B * T  # `CLIP_PER_ROW * ROWS`: the loss is summed over the rows
EPS = 1e-5
MODEL_DIR = CACHE.parent / f"nanolm2-d{D}-l{LAYERS}-v{VOCAB}-bench"
H = D // DH

corpus = Corpus()
leaves = [torch.tensor(a, requires_grad=True) for a in read_checkpoint(MODEL_DIR / "gpt_init.ckpt")]
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
COS, SIN = torch.cos(_theta), torch.sin(_theta)


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


def loss(x, y):
    h = tok[x.reshape(-1)]
    for p in blocks:
        h = block(h, p)
    return F.cross_entropy(rms_norm(h, nf) @ tok.T, y.reshape(-1), reduction="sum")


def mean_loss(tokens, seed, first, batches):
    with torch.no_grad():
        total = sum(loss(*map(torch.from_numpy, batch(tokens, seed, s))).item() for s in range(first, first + batches))
    return total / (batches * B * T)


def validation(batches):
    return mean_loss(corpus.val, VAL_SEED, 0, batches)


def learning_rate(s, base):
    if s < WARMUP:
        return base * (s + 1) / WARMUP
    t = (s - WARMUP) / (DECAY_STEPS - WARMUP)
    return base * (1.0 - 0.9 * t) if t < 1.0 else base * 0.1


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
# way: from one report to the next, the round's training steps plus the losses reported at its end.
# As in cleave, the total (`elapsed`) counts from before the first report, the rounds' minutes
# from right after it.
begin = time.perf_counter()
print(f"step 0: validation {corpus.per_token(validation(4))}, learning rate {learning_rate(0, LR):.6f}")
start = last = time.perf_counter()
for r in range(ROUNDS):
    for k in range(PER_ROUND):
        s = r * PER_ROUND + k
        x, y = map(torch.from_numpy, batch(corpus.train, TRAIN_SEED, s))
        for p in params:
            p.grad = None
        loss(x, y).backward()
        torch.nn.utils.clip_grad_norm_(params, CLIP)
        for g in adamw.param_groups:
            g["lr"] = learning_rate(s, LR)
        adamw.step()
        muon_step(learning_rate(s, MUON_LR))
    step = (r + 1) * PER_ROUND
    # The round's last four training batches, just trained on, as `gpt_train_loss` does.
    train = mean_loss(corpus.train, TRAIN_SEED, step - 4, 4)
    val = validation(4)
    now = time.perf_counter()
    print(
        f"step {step}: train {train:.4f}, validation {corpus.per_token(val)}, learning rate {learning_rate(step, LR):.6f}, "
        f"{(now - last) * 1000 / PER_ROUND:.0f} ms/step, {(now - start) / 60:.1f} min elapsed",
        flush=True,
    )
    last = now
nats = validation(20)
print(f"transformer: {corpus.per_token(nats)}")
print(f"elapsed: {time.perf_counter() - begin:.2f}s")
