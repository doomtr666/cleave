// The Vulkan backend's tuned products (`cleave-tile-matmuls-for-gpu`), written
// by `bench/gpu_matmul/tune.rs`: one line per product shape,
// `m n k a_k_contiguous b_k_contiguous bm bn bk tm tn split vec`, the fastest
// configuration measured for it. Included in `gpu.cpp` as a string.
// Measured on an NVIDIA GeForce RTX 4070 SUPER (2026-10-14).
R"(
1024 768 768 1 0 64 64 16 8 8 1 3
1024 2048 768 1 0 64 64 16 8 8 0 3
1024 768 2048 1 0 64 64 16 8 8 4 3
1024 4096 768 1 1 128 128 16 8 8 1 0
1024 768 4096 1 0 64 64 16 8 8 4 3
4096 768 1024 0 0 64 64 16 8 8 0 3
768 768 1024 0 0 64 64 16 8 8 2 1
768 2048 1024 0 0 64 64 16 8 8 2 1
1024 768 2048 1 1 128 128 32 8 8 1 0
1024 1024 64 1 1 64 64 16 8 8 0 1
1024 64 1024 1 0 64 64 16 8 4 8 1
768 768 768 1 1 64 64 16 8 8 2 1
768 768 2048 1 1 128 128 32 8 8 0 0
)"
