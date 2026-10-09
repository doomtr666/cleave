# Building cleave

Two things have to be true before `cargo build` works: an LLVM 22 + MLIR +
openmp toolchain, and `cargo` told where it is (`CLEAVE_LLVM_PREFIX`).
cleave talks to MLIR through its own C API (`cleave-mlir-shim`, compiled
with `cc` against that toolchain and linking it; `doc/plan-mlir-shim.md`):
no `melior`, `mlir-sys`, `tblgen`, bindgen or libclang.

## 1. Prerequisites

- **Visual Studio Build Tools**, MSVC toolset + a Windows SDK + the MASM
  component (`ml64.exe` — openmp's own runtime has real `.asm` sources).
  The "Desktop development with C++" workload covers all of this.
- **Rust** (stable, via `rustup`).
- **git**.

Everything below runs inside a Developer Command Prompt/PowerShell (i.e.
after `vcvarsall.bat`/`vcvars64.bat`, or via VS's own "Developer PowerShell"
shortcut) — `cl`/`link`/`lib`/`ml64` need to already be on `PATH`.

## 2. Getting the toolchain

```powershell
.\scripts\setup-toolchain.ps1
```

Downloads the prebuilt LLVM/clang/MLIR/openmp toolchain
[`cleave-llvm-redist`](https://github.com/doomtr666/cleave-llvm-redist)
publishes (its own CI builds it from source, `llvmorg-22.1.0`, with the
exact CMake configuration this project needs — see §5 below if you want
that configuration for building it yourself instead), caches it under
`target\llvm-mlir-22` (not a per-user system-drive cache — deliberately, so
a 1GB+ download lands on whichever drive this checkout/`target` already
lives on; `cargo clean` deletes it along with everything else under
`target`, so it re-downloads on the next build after that, an accepted
trade-off), and writes `.cargo/config.toml`
(gitignored, machine-specific — see `.cargo/config.toml.example` for the
tracked explanation of what it contains) with `CLEAVE_LLVM_PREFIX`, which
`cleave-mlir-shim`'s build script picks up. Idempotent:
re-running it is a no-op once the pinned version (`ci/toolchain-version.txt`)
is already cached — re-run it after that file changes, or pass `-Force` to
redownload regardless.

Already have your own LLVM/MLIR 22 build (from §5 below, or any other real
install with the right projects enabled)? Point at it directly instead, no
download:

```powershell
.\scripts\setup-toolchain.ps1 -ExistingPrefix C:\path\to\your\llvm-mlir-22
```

Windows/MSVC only today — `cleave-llvm-redist`'s own build workflow is
Windows-only so far (a Linux build is planned; once it exists, the script
picks the right asset for the host platform automatically, no separate
command to remember).

## 3. How cleave reaches MLIR

`cleave-mlir-shim` holds everything that touches MLIR and LLVM: its C++
(`cpp/shim.cpp`: the target and code generation, cleave's passes, the
pipeline runner) and the Rust side (`src/mlir`: MLIR's C API declared by
hand, and the IR API `mlir_lower.rs` builds with). Its `build.rs` compiles
the C++ with the same settings the toolchain was built with (release CRT,
no RTTI, no exceptions) and links every `MLIR*` library of the prefix, the
LLVM components `llvm-config --libnames` lists, and the system libraries.

## 4. Building and testing cleave itself

Ordinary Cargo from here:

```sh
RUST_MIN_STACK=67108864 cargo build --release
RUST_MIN_STACK=67108864 cargo test --release
```

`RUST_MIN_STACK` — not optional, confirmed directly, not a defensive
habit: `cargo test`'s own worker threads get an ordinary, small default
stack regardless of `--release`, and real tests overflow it (deep
e-graph/CPS recursion, the same class of depth `cleave-build`'s own
dedicated 1GB build thread and `main.rs`'s own 1GB main-thread stack
already exist to give the compiler itself) overflows it without this set —
on a local debug build and in CI's own `--release` run alike. `.github/
workflows/ci.yml` sets this on its own `cargo test` step already; nothing
gives you the same thing automatically outside CI, set it yourself.

`cargo build --workspace`/`cargo test --workspace` also walks
`examples/digits-interop`/`examples/mnist-interop` — real network access
(MNIST download) and multi-minute training runs, deliberately excluded
from `Cargo.toml`'s own `default-members` (see its own comment there), so
run those on purpose, not by habit.

## 5. Building the toolchain from source (fallback / how the release itself is built)

Not needed for ordinary development — §2's script is the fast path. This
section is what `cleave-llvm-redist`'s own CI actually runs (the reference
implementation if anything here goes stale), useful if you want a from-
source build for your own reasons (a custom LLVM patch, an architecture
`cleave-llvm-redist` doesn't publish for yet, ...).

Needs **CMake** and **Ninja** in addition to §1's prerequisites. Pinned
source:

```sh
git clone --branch llvmorg-22.1.0 --depth 1 https://github.com/llvm/llvm-project.git
```

[`ci/llvm-cmake-flags.txt`](../ci/llvm-cmake-flags.txt) is the single source
of truth for the CMake configuration (`cleave-llvm-redist`'s own
`cleave-toolchain-cmake-flags.txt` is a hand-kept copy of this same file —
that repo has no dependency on this one, so it can't just read it
directly). Its flags, and why each one is there:

| Flag | Why |
|---|---|
| `CMAKE_BUILD_TYPE=Release` | Optimized codegen — this project cares about the generated code's own runtime performance, not just compiling the toolchain fast. |
| `LLVM_ENABLE_PROJECTS=clang;mlir;openmp` | `mlir` is the real target; `openmp` backs `cleave`'s own OpenMP parallelization (`cleave/src/pipeline.rs`, `--openmp`/`CodegenOptions::openmp`). `clang` is included even though this project never calls it directly — openmp's own in-tree build unconditionally wires its optional lit-test targets (`check-openmp`/etc.) to a real `clang` target (confirmed directly against this exact LLVM tag — no `-D` flag can skip this, `openmp/cmake/OpenMPTesting.cmake`'s own `ENABLE_CHECK_TARGETS` is a plain variable, unconditionally reset on every configure, not a cache variable). Building `clang` for real satisfies that dependency honestly instead of patching LLVM's own source to work around it. |
| `LLVM_ENABLE_ASSERTIONS=ON` | Real correctness value, confirmed unrelated to this project's own compile-time issues (`doc/backlog.md`'s own "L'hypothèse LLVM_ENABLE_ASSERTIONS était une fausse piste" item — root-caused and fixed elsewhere, not by disabling this). |
| `LLVM_ENABLE_RTTI=OFF` | LLVM/MLIR's own default; `cleave-mlir-shim` is compiled the same way (`/GR-`). |
| `LLVM_TARGETS_TO_BUILD=Native` | Only the host's own architecture — cleave's own reference backend is CPU (`doc/hld.md`), no cross-compilation target needed today. |
| `LLVM_OPTIMIZED_TABLEGEN=OFF` | Matches this project's own dev toolchain; `ON` is a real, untried lever if a from-scratch build ever needs to be faster (`ci/llvm-cmake-flags.txt`'s own build ballooned once `clang` was added). |
| `LLVM_INSTALL_UTILS=OFF` | Not needed — this project only ever links against the installed libraries/headers, never runs LLVM's own dev utilities. |
| `LLVM_ENABLE_DIA_SDK=OFF` | Needs the ATL optional VS component, not installed on every toolset, and irrelevant to MLIR/openmp anyway. |

```powershell
cmake -S llvm-project\llvm -B llvm-project\build -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DLLVM_ENABLE_PROJECTS=clang;mlir;openmp `
  -DLLVM_ENABLE_ASSERTIONS=ON `
  -DLLVM_ENABLE_RTTI=OFF `
  -DLLVM_TARGETS_TO_BUILD=Native `
  -DLLVM_OPTIMIZED_TABLEGEN=OFF `
  -DLLVM_INSTALL_UTILS=OFF `
  -DLLVM_ENABLE_DIA_SDK=OFF `
  -DCMAKE_INSTALL_PREFIX=C:\llvm-mlir-22
cmake --build llvm-project\build --target install
```

Real, from-scratch time isn't small — this includes a full `clang` build,
not just MLIR/openmp. Budget real time (hours, not minutes) — this is
exactly the cost §2's prebuilt download exists to spare ordinary
development from paying repeatedly.

Once installed, point cleave at it with `scripts/setup-toolchain.ps1
-ExistingPrefix C:\llvm-mlir-22` (§2 above) rather than setting
`CLEAVE_LLVM_PREFIX` by hand.
