# Toward a first release (v0.1)

What the first public version of cleave should be, what it must contain, and in what order to get
there. A compass, not a deadline: the release isn't urgent, and the work toward it should stay
enjoyable. The point of this file is to keep the heading while doing the interesting parts.

## What cleave is, and how v0.1 should present it

cleave is a **general-purpose scientific language**: CFD, N-body/SPH, linear-algebra-heavy simulation
and ML training sit on the same footing (`README.md`, `doc/hld.md`). It is **not an ML framework**. It
does ML because a training loop is one more algebra-driven numerical computation, handled by the same
machinery as everything else: the e-graph, automatic differentiation, native code generation.

v0.1 should show exactly that: one language, several kinds of science, with ML as a strong example
rather than the identity.

## The showcase

Two demonstrations, chosen because each one would be convincing on its own:

- **A small character-level language model (nanoLM)**, nanoGPT-style: a few transformer layers, trained
  from scratch on a small text corpus, generating text at the end, with a line-for-line PyTorch twin for
  speed and accuracy comparison (as `bench/mnist-pytorch` does for MNIST). It stresses patterns MNIST
  never touches (embeddings, attention, softmax, LayerNorm, residual connections, Adam); the plan is in
  `doc/backlog.md`, "ML roadmap after MNIST".
- **cleave-cast, a physics-informed neural network (PINN) for weather prediction.** The demo that best
  matches the positioning: a neural network whose loss contains the governing physical equations, so it
  needs derivatives of the network with respect to its *inputs* (space, time) as well as gradients with
  respect to its weights — differential operators and learning in the same program, which is precisely
  where a general scientific language with built-in differentiation should beat a pure ML stack.

The existing examples (convex hull, Mandelbrot, linear regression, complex numbers, MNIST) round the
showcase out on the non-ML side.

## Minimum content of v0.1

Someone who has never seen cleave can install it, follow the guide, write a program that works, and
use cleave from a Rust project as easily as any other crate. An **alpha, proof-of-concept** release is
fine: known limitations get listed in the release notes rather than fixed at all costs, and bugs are
fixed opportunistically along the way rather than as a gate.

1. **CI green** on every supported platform.
2. **No crash on ordinary code.** One known offender today: a matmul whose row count isn't a multiple
   of 8 fails to compile. The `error: NYI: non-trivial layout map` diagnostics printed by builds with a
   matmul are noise, not a crash, and come from a lowering LLVM/MLIR doesn't implement, so anything
   short of avoiding that lowering would be a workaround: a known limitation, not a v0.1 requirement.
3. **Readable ML code.** The `nn` library and the MNIST example should read about as plainly as their
   PyTorch equivalent, without making the user fight generics and const generics
   (`doc/backlog.md`, "The `nn` library and the MNIST kernel are far harder to read and write...").
4. **Audit and polish.** Remove dead options, debug-only environment variables, leftover files and
   experiments; move finished entries out of `doc/backlog.md`; keep the user guide in sync with the
   language.
5. **A standalone, easy-to-use package.**
   - Done: cleave talks to MLIR only through its own C API (`cleave-mlir`, its C++ shim built with
     cleave against the prebuilt LLVM/MLIR; [`plan-mlir-shim.md`](plan-mlir-shim.md)). The two
     points below rest on it.
   - The `cleave` compiler ships as a prebuilt binary per platform (GitHub releases): the standalone
     tool, usable on its own.
   - Using cleave from Rust feels like using any crate (the experience `pest` gives): add `cleave-build`
     and `cleave-rt` as dependencies, point `build.rs` at a `.cleave` file, done. `cleave-build` fetches
     the matching prebuilt compiler instead of compiling LLVM/MLIR-linked code inside every user
     project, with an override for working on cleave itself. The executable it produces finds its
     runtime libraries (OpenMP, OpenBLAS) without manual setup.
6. **Linux support**, alongside Windows. The code is ready for it: the shim's build uses each
   compiler's spelling of its flags and links LLVM/MLIR by `llvm-config`, the OpenMP runtime's
   name and place come from one module (`cleave::toolchain`), OpenBLAS is loaded with `dlopen` off
   Windows, debug info is DWARF there (CodeView on Windows only), and executables find `libomp`
   through an rpath. Left: a prebuilt LLVM/MLIR and OpenBLAS for Linux (a CI leg of
   `cleave-llvm-redist`), the setup scripts (PowerShell today; `pwsh` runs on Linux, or a shell
   twin), a Linux CI job, and running the test suite there. Local testing through WSL2, which runs
   on the same CPU, so performance numbers stay comparable.
7. **The showcase**: nanoLM at least; cleave-cast if it's ready.

## Not required for v0.1

- Good multi-thread scaling (cleave already beats PyTorch on MNIST: 1.7x single-threaded, and 6.1 s on
  4 threads versus 7.8 s for PyTorch's best configuration on 8).
- The GPU (Vulkan) backend, macOS, ARM.

## Suggested order

Interleave the fun parts with the release work rather than finishing one before starting the other:

1. CI green.
2. `nn` ergonomics, by rewriting MNIST as simply as the compiler allows.
3. Softmax + cross-entropy, then nanoLM — which doubles as the second test of the new `nn` API.
4. Packaging and Linux (one release pipeline for both).
5. Audit and polish; release notes with known limitations; tag v0.1 (alpha).
6. cleave-cast, before or after the tag depending on how it goes.

Bugs found along the way get fixed when they're in the path, not as a separate phase.
