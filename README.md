# cleave

A converged language for HPC (high-performance computing) — CFD, N-body/SPH,
linear-algebra-heavy simulation, and ML/neural-network training all sit on
the same footing. ML is one HPC workload among many, not a distinct target
needing special-casing: a training loop is just another algebra-driven
numerical computation, handled by the same extensibility/optimization
machinery as any other domain.

Pipeline: a Pest grammar → surface AST → desugars to a functional core (CPS)
→ an e-graph (`egg`) for symbolic/algebraic optimization → cost-driven
extraction → progressive lowering to MLIR → target-specific codegen. See
[`doc/hld.md`](doc/hld.md) for the full design, [`doc/user_guide.md`](doc/user_guide.md)
for the language itself, and [`doc/backlog.md`](doc/backlog.md)/
[`doc/backlog-done.md`](doc/backlog-done.md) for what's real today versus
still open.

Project values: open source, no vendor lock-in. Reference hardware targets
are **CPU** (today) and **Vulkan Compute** (planned, via MLIR's `spirv`
dialect) — deliberately not a CUDA-only path.

## Building

Needs a real LLVM 22 + MLIR + openmp toolchain, `cargo` told where it is,
and `mlir-sys` resolved to this project's own fork (the unpatched
crates.io release doesn't link on Windows/MSVC at all) — see
**[`doc/building.md`](doc/building.md)** for the full procedure and why the
fork exists. Short version:

```powershell
.\scripts\setup-toolchain.ps1
cargo build --release
cargo test --release
```

The script downloads the prebuilt toolchain
[`cleave-llvm-redist`](https://github.com/doomtr666/cleave-llvm-redist)
publishes and points `cargo` at it automatically — no manual env vars, no
building LLVM from source (unless you want to; `doc/building.md` covers
that path too).

`cargo build --workspace`/`cargo test --workspace` also walks
`examples/digits-interop`/`examples/mnist-interop` — real network access
(MNIST download) and multi-minute training runs, deliberately excluded from
`default-members` (see `Cargo.toml`'s own comment), so run those on
purpose, not by habit.

## Platform

Windows/MSVC only today — the toolchain, `cleave-rt`, and the `mlir-sys`
fork this depends on are all Windows-specific work with no equivalent
upstream CI coverage (`mlir-sys`'s own CI is Linux/macOS only).
