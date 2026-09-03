"""PyTorch CPU baseline matching `examples/mnist-interop/src/kernel.cleave`
exactly, for a direct perf/accuracy comparison against cleave -- same
architecture, same init formulas, same loss shape, same optimizer, same
hyperparameters, same (unshuffled, sequential) batching. Deliberately does
*not* use `nn.Linear`/`nn.CrossEntropyLoss`/`torch.optim.SGD`'s own defaults
where those would silently diverge from cleave's own math (see comments at
each such point below) -- the point of this script is a fair comparison,
not "idiomatic PyTorch".

Architecture (`kernel.cleave`): 784 -> 512 -> 256 -> 128 -> 10, ReLU between
every hidden layer, no final activation (raw logits) -- plain sum-of-squares
regression against a one-hot target, not softmax+cross-entropy.

Run via poetry (keeps this fully out of your global Python install):
    cd bench/mnist-pytorch
    poetry install
    poetry run python mnist_bench.py
"""

import argparse
import gzip
import io
import math
import time
import urllib.request
from pathlib import Path

import torch

MIRROR = "https://raw.githubusercontent.com/fgnt/mnist/master"
PIXELS_PER_IMAGE = 28 * 28

# Same cache directory `examples/mnist-interop/src/data.rs` already uses --
# reuses the same on-disk IDX files instead of downloading a second copy.
DEFAULT_CACHE_DIR = (
    Path(__file__).resolve().parents[2] / "examples" / "mnist-interop" / ".cache"
)


def fetch_cached(cache_dir: Path, name: str) -> bytes:
    dest = cache_dir / name
    if dest.exists():
        return dest.read_bytes()
    cache_dir.mkdir(parents=True, exist_ok=True)
    url = f"{MIRROR}/{name}.gz"
    print(f"mnist_bench: downloading {url} ...")
    with urllib.request.urlopen(url) as resp:
        compressed = resp.read()
    raw = gzip.decompress(compressed)
    dest.write_bytes(raw)
    return raw


def parse_idx3_images(raw: bytes) -> torch.Tensor:
    magic = int.from_bytes(raw[0:4], "big")
    assert magic == 0x0000_0803, "not a real IDX3 image file (bad magic)"
    n = int.from_bytes(raw[4:8], "big")
    rows = int.from_bytes(raw[8:12], "big")
    cols = int.from_bytes(raw[12:16], "big")
    assert rows * cols == PIXELS_PER_IMAGE, f"expected 28x28 images, got {rows}x{cols}"
    data = raw[16:]
    assert len(data) == n * PIXELS_PER_IMAGE, "truncated IDX3 file"
    # Normalized to [0.0, 1.0] -- same normalization as `data.rs::parse_idx3_images`.
    buf = torch.frombuffer(bytearray(data), dtype=torch.uint8).float() / 255.0
    return buf.view(n, PIXELS_PER_IMAGE)


def parse_idx1_labels(raw: bytes) -> torch.Tensor:
    magic = int.from_bytes(raw[0:4], "big")
    assert magic == 0x0000_0801, "not a real IDX1 label file (bad magic)"
    n = int.from_bytes(raw[4:8], "big")
    data = raw[8:]
    assert len(data) == n, "truncated IDX1 file"
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).long()


def load_split(cache_dir: Path, images_name: str, labels_name: str):
    pixels = parse_idx3_images(fetch_cached(cache_dir, images_name))
    labels = parse_idx1_labels(fetch_cached(cache_dir, labels_name))
    assert pixels.shape[0] == labels.shape[0]
    return pixels, labels


def one_hot(labels: torch.Tensor, num_classes: int = 10) -> torch.Tensor:
    return torch.nn.functional.one_hot(labels, num_classes).float()


# ---------------------------------------------------------------- Init, exactly matching stdlib/nn/nn.cleave

def xavier_uniform(fan_in: int, fan_out: int, generator: torch.Generator) -> torch.Tensor:
    # `Init<Tensor<T,In,Out>>::xavier` (nn.cleave): uniform(-limit, limit),
    # limit = sqrt(6 / (fan_in + fan_out)).
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    w = torch.empty(fan_in, fan_out)
    w.uniform_(-limit, limit, generator=generator)
    return w.requires_grad_(True)


def he_normal(fan_in: int, fan_out: int, generator: torch.Generator) -> torch.Tensor:
    # `Init<Tensor<T,In,Out>>::he` (nn.cleave): normal(0, std), std = sqrt(2 / fan_in).
    std = math.sqrt(2.0 / fan_in)
    w = torch.empty(fan_in, fan_out)
    w.normal_(0.0, std, generator=generator)
    return w.requires_grad_(True)


def zero_bias(fan_out: int) -> torch.Tensor:
    # `Init<Dense<T,In,Out>>`: `b: Ring::zero()` -- bias always starts at
    # zero, *not* `nn.Linear`'s own default (uniform based on fan_in).
    return torch.zeros(fan_out, requires_grad=True)


class Net:
    """Plain parameter tensors, not `nn.Module`/`nn.Linear` -- keeps the
    weight layout as `(In, Out)` exactly matching `Tensor<T,In,Out>`
    (`x @ W`, not `nn.Linear`'s own transposed `(Out, In)` storage), and
    keeps every init/update formula explicit and auditable against
    `nn.cleave`/`optim.cleave` line by line rather than trusting a
    framework default to happen to match.
    """

    def __init__(self, generator: torch.Generator):
        self.w1, self.b1 = he_normal(784, 512, generator), zero_bias(512)
        self.w2, self.b2 = he_normal(512, 256, generator), zero_bias(256)
        self.w3, self.b3 = he_normal(256, 128, generator), zero_bias(128)
        self.w4, self.b4 = xavier_uniform(128, 10, generator), zero_bias(10)

    def params(self):
        return [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3, self.w4, self.b4]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = torch.relu(x @ self.w1 + self.b1)
        h2 = torch.relu(h1 @ self.w2 + self.b2)
        h3 = torch.relu(h2 @ self.w3 + self.b3)
        return h3 @ self.w4 + self.b4  # raw logits, no final activation


def sgd_step(params, lr: float):
    # Plain `Sgd(lr)` (`optim.cleave`): `w -= lr * grad`, no momentum, no
    # weight decay, no dampening -- hand-rolled rather than
    # `torch.optim.SGD` to guarantee no default-flag drift from that exact
    # shape.
    with torch.no_grad():
        for p in params:
            p -= lr * p.grad
            p.grad = None


def evaluate(net: Net, x_test: torch.Tensor, labels_test: torch.Tensor) -> float:
    with torch.no_grad():
        pred = net.forward(x_test)
        predicted = pred.argmax(dim=1)
        correct = (predicted == labels_test).sum().item()
    return correct / labels_test.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    ap.add_argument(
        "--debug-cap",
        type=int,
        default=None,
        help="truncate the training set (mirrors cleave's own MNIST_DEBUG_CAP) for a quick smoke run",
    )
    args = ap.parse_args()

    print(f"torch {torch.__version__}, CPU threads: {torch.get_num_threads()}")

    x_train, y_train_labels = load_split(
        args.cache_dir, "train-images-idx3-ubyte", "train-labels-idx1-ubyte"
    )
    x_test, y_test_labels = load_split(
        args.cache_dir, "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"
    )
    if args.debug_cap is not None:
        x_train, y_train_labels = x_train[: args.debug_cap], y_train_labels[: args.debug_cap]
    y_train = one_hot(y_train_labels)

    generator = torch.Generator().manual_seed(args.seed)
    net = Net(generator)

    n = x_train.shape[0]
    num_batches = n // args.batch_size  # no partial-batch remainder, matching kernel.cleave
    # `loss = sum(err*err)` over the whole batch (not mean) means the
    # gradient magnitude scales with batch size -- `train_and_evaluate`
    # compensates by dividing `lr` by the batch size once, at the call
    # site, not by changing `sum` itself. Mirrored here identically.
    lr = args.lr / args.batch_size

    start = time.perf_counter()
    for epoch in range(args.epochs):
        epoch_start = time.perf_counter()
        for s in range(num_batches):
            lo, hi = s * args.batch_size, (s + 1) * args.batch_size
            x, y = x_train[lo:hi], y_train[lo:hi]
            pred = net.forward(x)
            err = pred - y
            loss = (err * err).sum()
            loss.backward()
            sgd_step(net.params(), lr)
        epoch_elapsed = time.perf_counter() - epoch_start
        print(f"Epoch={epoch}  ({epoch_elapsed:.2f}s)")
    total_elapsed = time.perf_counter() - start

    accuracy = evaluate(net, x_test, y_test_labels)
    print(f"test accuracy: {accuracy}")
    print(f"elapsed: {total_elapsed:.2f}s ({total_elapsed / args.epochs:.2f}s/epoch)")


if __name__ == "__main__":
    main()
