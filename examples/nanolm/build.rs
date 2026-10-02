// Compiles `src/kernel.cleave` (see `examples/mnist-interop/build.rs` for the options).
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
