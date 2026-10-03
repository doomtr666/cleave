"""Twin of `train_gpt` in `examples/nanolm/src/kernel.cleave`: the same transformer (token and
position embeddings of width `D`, `LAYERS` pre-LayerNorm blocks with causal attention over heads
`DH` wide and a `D -> 4D -> D` GELU MLP, a final LayerNorm, a dense head to 104 logits), from the
same initial weights (`gpt_init.ckpt`, written by cleave's `bench` mode in its own directory, apart
from the real run's checkpoints), on the same
batches, with the same Adam, learning-rate schedule and summed loss. The sizes must match the
kernel's `define`s. The validation losses must match cleave's.

    cargo run --release -p nanolm -- bench 0 <rounds> <steps_per_round>   (writes gpt_init.ckpt)
    poetry run python gpt.py 0 <rounds> <steps_per_round>

The same arguments as cleave's `bench` mode; the first step must be 0, the twin doesn't resume.
"""

import math
import sys
import time

import torch
import torch.nn.functional as F

from data import CACHE, TRAIN_SEED, VAL_SEED, Corpus, batch, read_checkpoint

arg = lambda i, default: int(sys.argv[i]) if len(sys.argv) > i else default
FIRST, ROUNDS, PER_ROUND = arg(1, 0), arg(2, 10), arg(3, 100)
if FIRST != 0:
    sys.exit("the twin starts from gpt_init.ckpt: the first step must be 0 (it doesn't resume)")
LR, B, T, D, DH, LAYERS = 0.001, 32, 128, 256, 64, 6
WARMUP, DECAY_STEPS = 200, 20000
MODEL_DIR = CACHE.parent / f"gpt-d{D}-l{LAYERS}-bench"
H = D // DH

corpus = Corpus()
leaves = iter(torch.tensor(a, requires_grad=True) for a in read_checkpoint(MODEL_DIR / "gpt_init.ckpt"))
take = lambda n: [next(leaves) for _ in range(n)]
tok, pos = take(2)
# Per block, in `Block`'s field order: ln1 g, b; wq, wk, wv, wo (w, b each); ln2 g, b; fc, proj (w, b).
blocks = [take(16) for _ in range(LAYERS)]
lnf_g, lnf_b, head_w, head_b = take(4)
params = [tok, pos, *[p for blk in blocks for p in blk], lnf_g, lnf_b, head_w, head_b]
positions = torch.arange(B * T) % T


def layer_norm(x, g, b):
    return F.layer_norm(x, (x.shape[-1],), g[0], b[0], eps=1e-5)


def attention(q, k, v):
    split = lambda m: m.reshape(B, T, H, DH).permute(0, 2, 1, 3)
    s = split(q) @ split(k).transpose(-1, -2) / math.sqrt(DH)
    s = s.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool), 1), float("-inf"))
    return (torch.softmax(s, -1) @ split(v)).permute(0, 2, 1, 3).reshape(B * T, D)


def block(x, p):
    ln1_g, ln1_b, wq, bq, wk, bk, wv, bv, wo, bo, ln2_g, ln2_b, w1, b1, w2, b2 = p
    h = layer_norm(x, ln1_g, ln1_b)
    x = x + attention(h @ wq + bq, h @ wk + bk, h @ wv + bv) @ wo + bo
    return x + F.gelu(layer_norm(x, ln2_g, ln2_b) @ w1 + b1, approximate="tanh") @ w2 + b2


def loss(x, y):
    h = tok[x.reshape(-1)] + pos[positions]
    for p in blocks:
        h = block(h, p)
    return F.cross_entropy(layer_norm(h, lnf_g, lnf_b) @ head_w + head_b, y.reshape(-1), reduction="sum")


def mean_loss(tokens, seed, first, batches):
    with torch.no_grad():
        total = sum(loss(*map(torch.from_numpy, batch(tokens, seed, s))).item() for s in range(first, first + batches))
    return total / (batches * B * T)


def validation(batches):
    return mean_loss(corpus.val, VAL_SEED, 0, batches)


def learning_rate(s):
    if s < WARMUP:
        return LR * (s + 1) / WARMUP
    t = (s - WARMUP) / (DECAY_STEPS - WARMUP)
    return LR * (1.0 - 0.9 * t) if t < 1.0 else LR * 0.1


opt = torch.optim.Adam(params, lr=LR, betas=(0.9, 0.999), eps=1e-8)
# The same lines as the cleave host's `round_done` (`examples/nanolm/src/main.rs`), timed the same
# way: from one report to the next, the round's training steps plus the losses reported at its end.
# As in cleave, the total (`elapsed`) counts from before the first report, the rounds' minutes
# from right after it.
begin = time.perf_counter()
print(f"step 0: validation {validation(4):.4f}, learning rate {learning_rate(0):.6f}")
start = last = time.perf_counter()
for r in range(ROUNDS):
    for k in range(PER_ROUND):
        s = r * PER_ROUND + k
        x, y = map(torch.from_numpy, batch(corpus.train, TRAIN_SEED, s))
        for g in opt.param_groups:
            g["lr"] = learning_rate(s)
        opt.zero_grad()
        loss(x, y).backward()
        opt.step()
    step = (r + 1) * PER_ROUND
    # The round's last four training batches, just trained on, as `gpt_train_loss` does.
    train = mean_loss(corpus.train, TRAIN_SEED, step - 4, 4)
    val = validation(4)
    now = time.perf_counter()
    print(
        f"step {step}: train {train:.4f}, validation {val:.4f}, learning rate {learning_rate(step):.6f}, "
        f"{(now - last) * 1000 / PER_ROUND:.0f} ms/step, {(now - start) / 60:.1f} min elapsed",
        flush=True,
    )
    last = now
nats = validation(20)
print(f"transformer: {nats:.4f} nats/char ({nats / math.log(2):.4f} bits/char)")
print(f"elapsed: {time.perf_counter() - begin:.2f}s")
