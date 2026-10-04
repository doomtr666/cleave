"""The bigram baseline of `examples/nanolm/src/kernel.cleave`, on the same batches: the same
number means the twin reads the same corpus and draws the same batches.

    python bigram.py      (numpy only)
"""

import math

import numpy as np

from data import TRAIN_SEED, VAL_SEED, VOCAB, Corpus, batch

TRAIN_BATCHES, VAL_BATCHES = 200, 50
V = VOCAB

corpus = Corpus()
counts = np.ones((V, V), dtype=np.float64)
for s in range(TRAIN_BATCHES):
    x, y = batch(corpus.train, TRAIN_SEED, s)
    np.add.at(counts, (x.ravel(), y.ravel()), 1.0)
logp = np.log(counts) - np.log(counts.sum(axis=1, keepdims=True))
total = 0.0
for s in range(VAL_BATCHES):
    x, y = batch(corpus.val, VAL_SEED, s)
    total -= logp[x.ravel(), y.ravel()].sum()
nats = total / (VAL_BATCHES * x.size)
x, _ = batch(corpus.train, TRAIN_SEED, 0)
print(f"first training window: {corpus.decode(x[0])!r}")
print(f"bigram baseline: {corpus.per_token(nats)}")
