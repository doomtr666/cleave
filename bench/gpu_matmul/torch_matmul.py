"""cuBLAS's time on nanoLM's product shapes, the reference for cleave's Vulkan kernels
(`bench/gpu_matmul/gpu_matmul.rs`): each product as PyTorch computes it on CUDA in f32 (TF32
off), with the same layouts (`matmul`, `matmul_transpose_a`, `matmul_transpose_b`), timed with
CUDA events, the least of several batches.

    cd bench/nanolm-pytorch; poetry run python ../gpu_matmul/torch_matmul.py
"""

import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# (name, m, n, k, layout): the product of an m x k by a k x n, A transposed ("tn"), B transposed
# ("nt") or neither ("nn"), as `bench/gpu_matmul/matmul.cleave` computes it.
SHAPES = [
    ("qkvo_fwd", 1024, 768, 768, "nn"),
    ("gate_up", 1024, 2048, 768, "nn"),
    ("down", 1024, 768, 2048, "nn"),
    ("head", 1024, 4096, 768, "nt"),
    ("head_dx", 1024, 768, 4096, "nn"),
    ("head_dw", 4096, 768, 1024, "tn"),
    ("dw_qkvo", 768, 768, 1024, "tn"),
    ("dw_up", 768, 2048, 1024, "tn"),
    ("dx_up", 1024, 768, 2048, "nt"),
    ("attn_qk", 1024, 1024, 64, "nt"),
    ("attn_pv", 1024, 64, 1024, "nn"),
    ("muon_xxt", 768, 768, 768, "nt"),
    ("muon_wide", 768, 768, 2048, "nt"),
]

device = torch.device("cuda")
print(f"{'shape':>10} {'m x n x k':>16} {'us':>8} {'GFLOP/s':>9}")
total_us, total_flops = 0.0, 0.0
for name, m, n, k, layout in SHAPES:
    a = torch.randn(k, m, device=device).T if layout == "tn" else torch.randn(m, k, device=device)
    b = torch.randn(n, k, device=device).T if layout == "nt" else torch.randn(k, n, device=device)
    for _ in range(10):
        torch.mm(a, b)
    best = float("inf")
    for _ in range(5):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(20):
            torch.mm(a, b)
        end.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end) * 1000 / 20)
    flops = 2.0 * m * n * k
    total_us += best
    total_flops += flops
    print(f"{name:>10} {f'{m}x{n}x{k}':>16} {best:>8.1f} {flops / best / 1e3:>9.0f}")
print(f"{'all':>10} {'':>16} {total_us:>8.1f} {total_flops / total_us / 1e3:>9.0f}")
