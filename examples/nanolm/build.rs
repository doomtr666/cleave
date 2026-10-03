// Compiles `src/kernel.cleave` (see `examples/mnist-interop/build.rs` for the options).
const OPT_LEVEL: u8 = 2;
const OPENMP: bool = false;
const INLINE: bool = true;
const DEBUG: bool = true;
// LLVM's own loop unrolling, on top of cleave's: see `CodegenOptions::llvm_loop_unroll`.
const LLVM_LOOP_UNROLL: bool = false;

fn main() {
    cleave_build::Build::new()
        .file("src/kernel.cleave")
        .opt_level(OPT_LEVEL)
        .openmp(OPENMP)
        .inline(INLINE)
        .debug_info(DEBUG)
        .llvm_loop_unroll(LLVM_LOOP_UNROLL)
        .target_cpu("native")
        .compile("kernel");
}
