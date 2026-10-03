// Compiles `src/kernel.cleave` -- see `examples/rust-interop-demo/build.rs`'s
// own identical comment for the mechanism.
//
// `OPENMP` below -- a plain `const`, not an env var read at build-script
// runtime (`doc/backlog.md`'s own "every real codegen gate migrated..."
// entry: the whole point of that refactor was exactly this, real, visible,
// CLI/`Build`-API-controlled options instead of an invisible, easy-to-
// forget `CLEAVE_*` env var, and a build script reading one back out just
// reintroduces the same problem one layer up). Flip this constant by hand
// and rebuild (`cargo build`, or `touch build.rs` first if cargo doesn't
// notice -- editing this file's own source already re-triggers the build
// script, no `cargo:rerun-if-env-changed` needed for a plain constant) for
// a genuine mono-thread comparison against PyTorch (`bench/mnist-pytorch/
// mnist_bench.py` run under `OMP_NUM_THREADS=1`) -- `false` turns off
// `CodegenOptions::openmp` entirely, not just "run the parallelized build
// with one thread": `pipeline.rs`'s own doc comment on that field is
// explicit that this skips the whole `--affine-parallelize`/`--convert-scf-
// to-openmp`/`--convert-openmp-to-llvm` stage, genuinely serial generated
// code, no OpenMP fork-join scaffolding present at all (not merely idle at
// runtime).
//
// This exact toggle is what surfaced a real, separate compiler bug (`doc/
// backlog-done.md`'s own "An AOT binary built with `--no-openmp`...
// genuinely crashed with a native stack overflow" entry has the full
// root-cause and fix story: a per-loop-iteration native-stack leak in
// `pipeline.rs::lower_to_llvm`, nothing to do with `mnist-interop`
// specifically) -- fixed there, confirmed by running the real training loop
// below to completion under this exact toggle afterward.
const OPT_LEVEL: u8 = 2;
const OPENMP: bool = false;
const INLINE: bool = true;
const DEBUG: bool = true;

fn main() {
    cleave_build::Build::new()
        .file("src/kernel.cleave")
        .opt_level(OPT_LEVEL)
        .openmp(OPENMP)
        .inline(INLINE)
        .debug_info(DEBUG)
        .target_cpu("native")
        .compile("kernel");
}
