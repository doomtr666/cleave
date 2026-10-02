"""Twin of `train_gpt` in `examples/nanolm/src/kernel.cleave`: the same transformer (token and
position embeddings of width 128, 4 pre-LayerNorm blocks with causal attention over 4 heads of 32
and a 128 -> 512 -> 128 GELU MLP, a final LayerNorm, a dense head to 104 logits), from the same
initial weights (`gpt_init.ckpt`, written by cleave), on the same batches, with the same Adam and
the same summed loss. The validation losses must match cleave's.

    cargo run --release -p nanolm -- gpt 0 <rounds> <steps_per_round>   (writes gpt_init.ckpt)
    poetry run python gpt.py <rounds> <steps_per_round>
"""

import math
import sys
import time

import torch
import torch.nn.functional as F

from data import CACHE, TRAIN_SEED, VAL_SEED, Corpus, batch, read_checkpoint

ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
PER_ROUND = int(sys.argv[2]) if len(sys.argv) > 2 else 100
LR, B, T, D, DH = 0.001, 32, 128, 128, 32
H = D // DH

corpus = Corpus()
leaves = iter(torch.tensor(a, requires_grad=True) for a in read_checkpoint(CACHE.parent / "gpt_init.ckpt"))
take = lambda n: [next(leaves) for _ in range(n)]
tok, pos = take(2)
# Per block, in `Block`'s field order: ln1 g, b; wq, wk, wv, wo (w, b each); ln2 g, b; fc, proj (w, b).
blocks = [take(16) for _ in range(4)]
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


def validation(batches):
    with torch.no_grad():
        total = sum(loss(*map(torch.from_numpy, batch(corpus.val, VAL_SEED, s))).item() for s in range(batches))
    return total / (batches * B * T)


opt = torch.optim.Adam(params, lr=LR, betas=(0.9, 0.999), eps=1e-8)
print(f"step 0, validation: {validation(4):.7f}")
start = time.perf_counter()
for r in range(ROUNDS):
    for k in range(PER_ROUND):
        x, y = map(torch.from_numpy, batch(corpus.train, TRAIN_SEED, r * PER_ROUND + k))
        opt.zero_grad()
        loss(x, y).backward()
        opt.step()
    print(f"step {(r + 1) * PER_ROUND}, validation: {validation(4):.7f}")
print(f"transformer: {validation(20):.4f} nats/char")
print(f"elapsed: {time.perf_counter() - start:.2f}s (training, with validation)")
