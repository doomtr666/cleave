# Building cleave

Three real, separate things have to be true before `cargo build` works at
all: a real LLVM 22 + MLIR + openmp toolchain built from source, `cargo`
told where it is, and `mlir-sys` resolved to this project's own fork (not
the unpatched crates.io release, which doesn't link on Windows/MSVC at
all). This doc covers all three, in order — `.github/workflows/ci.yml` runs
the exact same procedure on a clean runner, so it's also the reference
implementation if anything here goes stale.

## 1. Prerequisites

- **Visual Studio Build Tools**, MSVC toolset + a Windows SDK + the MASM
  component (`ml64.exe` — openmp's own runtime has real `.asm` sources).
  The "Desktop development with C++" workload covers all of this.
- **CMake** and **Ninja** (the generator this whole toolchain is built
  with — `LLVM_ENABLE_PROJECTS`/every other flag below assumes it).
- **Rust** (stable, via `rustup`).
- **git**.

Everything below runs inside a Developer Command Prompt/PowerShell (i.e.
after `vcvarsall.bat`/`vcvars64.bat`, or via VS's own "Developer PowerShell"
shortcut) — `cl`/`link`/`lib`/`ml64` need to already be on `PATH`.

## 2. Building LLVM + MLIR + openmp from source

No prebuilt Windows package exists for this exact combination (unlike
Homebrew on Linux/macOS, which `mlir-sys`'s own upstream CI already uses —
`brew install llvm@22` — Homebrew has no Windows story at all). Pinned
source:

```sh
git clone --branch llvmorg-22.1.0 --depth 1 https://github.com/llvm/llvm-project.git
```

[`ci/llvm-cmake-flags.txt`](../ci/llvm-cmake-flags.txt) is the single source
of truth for the CMake configuration — read by `.github/workflows/ci.yml`
directly (both for its own cache key and its own `cmake` invocation), so it
never drifts from what CI actually builds. Its flags, and why each one is
there:

| Flag | Why |
|---|---|
| `CMAKE_BUILD_TYPE=Release` | Optimized codegen — this project cares about the generated code's own runtime performance, not just compiling the toolchain fast. |
| `LLVM_ENABLE_PROJECTS=clang;mlir;openmp` | `mlir` is the real target; `openmp` backs `cleave`'s own OpenMP parallelization (`cleave/src/pipeline.rs`, `--openmp`/`CodegenOptions::openmp`). `clang` is included even though this project never calls it directly — openmp's own in-tree build unconditionally wires its optional lit-test targets (`check-openmp`/etc.) to a real `clang` target (confirmed directly against this exact LLVM tag — no `-D` flag can skip this, `openmp/cmake/OpenMPTesting.cmake`'s own `ENABLE_CHECK_TARGETS` is a plain variable, unconditionally reset on every configure, not a cache variable). Building `clang` for real satisfies that dependency honestly instead of patching LLVM's own source to work around it. |
| `LLVM_ENABLE_ASSERTIONS=ON` | Real correctness value, confirmed unrelated to this project's own compile-time issues (`doc/backlog.md`'s own "L'hypothèse LLVM_ENABLE_ASSERTIONS était une fausse piste" item — root-caused and fixed elsewhere, not by disabling this). |
| `LLVM_ENABLE_RTTI=OFF` | LLVM/MLIR's own default; `melior`/`mlir-sys` expect it. |
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
not just MLIR/openmp. Budget real time (hours, not minutes) the first time;
`.github/workflows/ci.yml` caches its own install output for exactly this
reason, keyed on the LLVM tag + `ci/llvm-cmake-flags.txt`'s own hash.

Point `mlir-sys`'s own build script at the result (see §4 below) via
`MLIR_SYS_220_PREFIX`/`TABLEGEN_220_PREFIX` — `C:\llvm-mlir-22` in the
example above, but any install prefix works.

## 3. Why cleave needs a forked `mlir-sys`

`cleave/Cargo.toml`'s own `[patch.crates-io]` points `mlir-sys` at
[`doomtr666/mlir-sys`](https://github.com/doomtr666/mlir-sys)'s own
`fix/windows-msvc-static-linking` branch, not the crates.io release —
three real, independent Windows/MSVC bugs in the unpatched crate, each
found by direct testing (a real `func.func` with `arith.addi`, built and
verified against a from-source LLVM/MLIR 22.1.0 install), none yet released
upstream:

1. **Enum signedness.** `bindgen` reflects Clang's own per-target enum
   underlying-type inference faithfully — the MLIR C API's enums come out
   `c_int` on `-pc-windows-msvc`, but `melior`'s own hand-written Rust
   source (developed/tested on Linux/macOS, where the same enums infer
   `c_uint`) hardcodes `u32` unconditionally — a real compile failure on
   Windows, not a runtime bug. Fixed by post-processing the generated
   bindings, normalizing the affected enum type aliases from `c_int` to
   `c_uint` (same 4-byte C ABI representation either way — only Rust's own
   signedness declaration changes) — scoped to `CARGO_CFG_TARGET_ENV ==
   "msvc"` specifically, so it's provably a no-op on Linux/macOS rather
   than just probably one.
2. **A doubled `.lib` suffix.** `llvm-config --system-libs` reports names
   already carrying their own `.lib` suffix on Windows (unlike Unix's bare
   `-lfoo` form) — passed through unchanged, `cargo:rustc-link-lib` appends
   the platform suffix a second time, producing a nonexistent
   `psapi.lib.lib` at link time.
3. **Static-library discovery assumed Unix naming.** The `MLIRCAPI*`
   libraries are never reported by `llvm-config --libnames` at all (a real,
   verified gap — zero CAPI entries) — only a disk-scan fallback finds
   them, but its own `starts_with("libMLIR")` check and its own static-lib
   name parser both assumed a `lib` prefix. Windows static libraries have
   no such prefix (`MLIRCAPIIR.lib`, never `libMLIRCAPIIR.lib` — confirmed
   directly: 0 of 389 real `MLIR*.lib` files in an actual install start
   with `lib`). Every `MLIR*` static library was silently unlinked on
   Windows; only whichever symbols a given program actually referenced
   surfaced as `LNK2019` errors, which is why this showed up as a handful
   of `MLIRCAPIIR`-specific unresolved symbols rather than a wall of them.

Referencing the fork's own branch via a real `git` dependency (not a local
`path`, which this project used until it broke on any machine other than
the one it was first developed on) means `cargo build` needs network access
to GitHub the first time it resolves dependencies, same as any other `git`
dependency.

## 4. Pointing cleave at your own toolchain

`mlir-sys`'s own build script (and `tblgen-rs`'s, a transitive dependency)
read these two environment variables — not committed anywhere in this repo
(a machine-specific path), set them yourself, once, in your own shell
profile:

```powershell
[Environment]::SetEnvironmentVariable("MLIR_SYS_220_PREFIX", "C:\llvm-mlir-22", "User")
[Environment]::SetEnvironmentVariable("TABLEGEN_220_PREFIX", "C:\llvm-mlir-22", "User")
```

(`220` is the naming convention `mlir-sys`/`tblgen-rs` use for "LLVM major
version 22, no minor" — not a typo, and not configurable to a different
number.) Open a new shell afterward so the variables actually take effect.

## 5. Building and testing cleave itself

Ordinary Cargo from here:

```sh
cargo build --release
cargo test --release
```

`cargo build --workspace`/`cargo test --workspace` also walks
`examples/digits-interop`/`examples/mnist-interop` — real network access
(MNIST download) and multi-minute training runs, deliberately excluded
from `Cargo.toml`'s own `default-members` (see its own comment there), so
run those on purpose, not by habit.
