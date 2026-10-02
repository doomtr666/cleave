"""Twin of `train_lm` in `examples/nanolm/src/kernel.cleave`: the same model (embedding 104 -> 64,
dense 64 -> 256, GELU (tanh form), dense 256 -> 104), from the same initial weights (the
`lm_init.ckpt` cleave writes), on the same batches, with the same Adam and the same summed loss.
The validation losses printed every 100 steps must match cleave's.

    cargo run --release -p nanolm        (writes lm_init.ckpt)
    poetry run python mlp.py
"""

import time

import torch
import torch.nn.functional as F

from data import CACHE, TRAIN_SEED, VAL_SEED, Corpus, batch, read_checkpoint

ROUNDS, LR = 10, 0.003

corpus = Corpus()
emb, w1, b1, w2, b2 = (torch.tensor(a, requires_grad=True) for a in read_checkpoint(CACHE.parent / "lm_init.ckpt"))
params = [emb, w1, b1, w2, b2]


def loss(x, y):
    h = F.gelu(emb[x.reshape(-1)] @ w1 + b1, approximate="tanh")
    return F.cross_entropy(h @ w2 + b2, y.reshape(-1), reduction="sum")


def validation(batches):
    with torch.no_grad():
        total = sum(loss(*map(torch.from_numpy, batch(corpus.val, VAL_SEED, s))).item() for s in range(batches))
    return total / (batches * 4096)


opt = torch.optim.Adam(params, lr=LR, betas=(0.9, 0.999), eps=1e-8)
print(f"step 0, validation: {validation(10):.7f}")
start = time.perf_counter()
for r in range(ROUNDS):
    for k in range(100):
        x, y = map(torch.from_numpy, batch(corpus.train, TRAIN_SEED, r * 100 + k))
        opt.zero_grad()
        loss(x, y).backward()
        opt.step()
    print(f"step {(r + 1) * 100}, validation: {validation(10):.7f}")
nats = validation(50)
print(f"embedding + MLP: {nats:.4f} nats/char")
print(f"elapsed: {time.perf_counter() - start:.2f}s (training, with validation)")
