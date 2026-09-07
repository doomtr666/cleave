// Compiles `src/kernel.cleave` -- see `examples/rust-interop-demo/build.rs`'s
// own identical comment for the mechanism.
//
// `CLEAVE_NO_OPENMP=1` -- a one-off knob for a genuine mono-thread
// comparison against PyTorch (`bench/mnist-pytorch/mnist_bench.py` run under
// `OMP_NUM_THREADS=1`): turns off `CodegenOptions::openmp` entirely, not
// just "run the parallelized build with one thread" -- `pipeline.rs`'s own
// doc comment on that field is explicit that this skips the whole
// `--affine-parallelize`/`--convert-scf-to-openmp`/`--convert-openmp-to-llvm`
// stage, genuinely serial generated code, no OpenMP fork-join scaffolding
// present at all (not merely idle at runtime). `cargo:rerun-if-env-changed`
// so toggling this and rebuilding actually re-runs this script -- Cargo
// doesn't know to invalidate a build script's own cached output just
// because an env var it reads changed, unlike a `cargo:rerun-if-changed`
// file dependency.
//
// This exact toggle is what surfaced a real, separate compiler bug (`doc/
// backlog-done.md`'s own "An AOT binary built with `--no-openmp`...
// genuinely crashed with a native stack overflow" entry has the full
// root-cause and fix story: a per-loop-iteration native-stack leak in
// `pipeline.rs::lower_to_llvm`, nothing to do with `mnist-interop`
// specifically) -- fixed there, confirmed by running the real training loop
// below to completion under this exact toggle afterward.
fn main() {
    println!("cargo:rerun-if-env-changed=CLEAVE_NO_OPENMP");
    let openmp = std::env::var("CLEAVE_NO_OPENMP").is_err();
    cleave_build::Build::new()
        .file("src/kernel.cleave")
        .openmp(openmp)
        .compile("kernel");
}
