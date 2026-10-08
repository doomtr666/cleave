// Compiles `src/kernel.cleave` (see `examples/mnist-interop/build.rs` for the options).
const OPT_LEVEL: u8 = 2;
const OPENMP: bool = false;
const INLINE: bool = true;
const DEBUG: bool = true;
// LLVM's own loop unrolling, on top of cleave's: see `CodegenOptions::llvm_loop_unroll`.
const LLVM_LOOP_UNROLL: bool = false;

/// The kernel's `define NAME: i32 = <literal>;`s the host shares (`data.rs`'s `B`, `T`, `VOCAB`),
/// written to `$OUT_DIR/sizes.rs`: one source for the sizes, the kernel's.
fn write_sizes() {
    let kernel = std::fs::read_to_string("src/kernel.cleave").expect("cannot read src/kernel.cleave");
    let define = |name: &str| -> usize {
        let prefix = format!("define {name}: i32 = ");
        kernel
            .lines()
            .find_map(|l| l.strip_prefix(&prefix)?.strip_suffix(';')?.trim().parse().ok())
            .unwrap_or_else(|| panic!("kernel.cleave has no `{prefix}<integer>;`"))
    };
    let sizes = format!(
        "pub const B: usize = {};\npub const T: usize = {};\npub const VOCAB: usize = {};\n",
        define("BATCH"),
        define("CONTEXT"),
        define("VOCAB")
    );
    let out = std::path::Path::new(&std::env::var("OUT_DIR").unwrap()).join("sizes.rs");
    std::fs::write(out, sizes).expect("cannot write sizes.rs");
}

fn main() {
    write_sizes();
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
