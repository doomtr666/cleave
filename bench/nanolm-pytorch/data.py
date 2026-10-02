"""The corpus and batches of `examples/nanolm`, read from the same cache and drawn the same way
(`examples/nanolm/src/data.rs`): run `cargo run --release -p nanolm` once first to build
`.cache/zola/{train,val}.bin`.

Batch `i` of a split holds `B` windows of `T + 1` characters starting at
`splitmix64(seed ^ (i * B + b)) % (len - T)`; the first `T` are the inputs, the last `T` the
targets.
"""

from pathlib import Path

import numpy as np

B = 32
T = 128
TRAIN_SEED = 0x5A01A_7A1
VAL_SEED = 0x5A01A_7A2

CACHE = Path(__file__).resolve().parents[2] / "examples" / "nanolm" / ".cache" / "zola"
M64 = (1 << 64) - 1


def splitmix64(x: int) -> int:
    z = (x + 0x9E3779B97F4A7C15) & M64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
    return z ^ (z >> 31)


assert splitmix64(0) == 0xE220A8397B1DCDAF and splitmix64(1) == 0x910A2DEC89025CC1


class Corpus:
    def __init__(self, cache: Path = CACHE):
        self.alphabet = (cache / "alphabet.txt").read_text(encoding="utf-8")
        self.train = np.fromfile(cache / "train.bin", dtype=np.uint8)
        self.val = np.fromfile(cache / "val.bin", dtype=np.uint8)

    def decode(self, ids) -> str:
        return "".join(self.alphabet[i] for i in ids)


def batch(text: np.ndarray, seed: int, i: int):
    """`(inputs, targets)`, two `[B, T]` int64 arrays."""
    windows = len(text) - T
    starts = [splitmix64(seed ^ (i * B + b)) % windows for b in range(B)]
    rows = np.stack([text[s : s + T + 1] for s in starts]).astype(np.int64)
    return rows[:, :-1], rows[:, 1:]


def read_checkpoint(path: Path) -> list:
    """The leaves of a cleave checkpoint (`cleave-rt/src/checkpoint.rs`), in order: a numpy array
    for a tensor, a Python number for a scalar."""
    dtypes = {1: np.float32, 2: np.float64, 3: np.int32, 4: np.int64}
    raw = path.read_bytes()
    assert raw[:8] == b"CLVCKPT\0", f"{path} is not a cleave checkpoint"
    assert int.from_bytes(raw[8:12], "little") == 1, "unsupported checkpoint version"
    pos, leaves = 12, []
    while pos < len(raw):
        tag, rank = raw[pos], raw[pos + 1]
        pos += 2
        dims = [int.from_bytes(raw[pos + 8 * k : pos + 8 * k + 8], "little", signed=True) for k in range(rank)]
        pos += 8 * rank
        dtype = np.dtype(dtypes[tag])
        count = int(np.prod(dims)) if dims else 1
        data = np.frombuffer(raw, dtype=dtype, count=count, offset=pos)
        pos += count * dtype.itemsize
        leaves.append(data.reshape(dims).copy() if dims else data[0].item())
    return leaves
