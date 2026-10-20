// Compiles `src/kernel.cleave` (see `examples/mnist-interop/build.rs` for the options).
const OPT_LEVEL: u8 = 2;
const OPENMP: bool = false;
const INLINE: bool = true;
const DEBUG: bool = true;
// LLVM's own loop unrolling, on top of cleave's: see `CodegenOptions::llvm_loop_unroll`.
const LLVM_LOOP_UNROLL: bool = false;

/// `--features small`: the model's sizes overridden (`Build::define`), the host's too
/// (`write_sizes`). Its checkpoints go to a directory of their own, named after the sizes.
const SMALL: &[(&str, &str)] =
    &[("WIDTH", "256"), ("LAYERS", "4"), ("HIDDEN", "704"), ("CONTEXT", "256")];

fn small() -> bool {
    std::env::var_os("CARGO_FEATURE_SMALL").is_some()
}

/// The kernel's `define NAME: i32 = <literal>;`s the host shares (`data.rs`'s `B`, `T`, `VOCAB`),
/// written to `$OUT_DIR/sizes.rs`: one source for the sizes, the kernel's (or `SMALL`'s).
fn write_sizes() {
    let kernel = std::fs::read_to_string("src/kernel.cleave").expect("cannot read src/kernel.cleave");
    let define = |name: &str| -> usize {
        if let Some((_, value)) = SMALL.iter().find(|(n, _)| small() && *n == name) {
            return value.parse().expect("an integer size");
        }
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

/// The kernel's `SPLIT`, its batch's micro-batches (`parallel_grad`), per backend: one per core on
/// the CPU, where they run as tasks in parallel (8); few on Vulkan, which has no host tasks and runs
/// them one after another, fewer dispatches for fewer parts (2: 2.4% faster than 8, 2026-10-15;
/// 4% faster than 1, whose activations are twice as large, 2026-10-20). The program doesn't choose a target's
/// tuning (`doc/backlog.md`, "`spawn` on the GPU"). `NANOLM_SPLIT=<n>` overrides it, to measure
/// another.
fn split(vulkan: bool) -> String {
    println!("cargo:rerun-if-env-changed=NANOLM_SPLIT");
    std::env::var("NANOLM_SPLIT").unwrap_or_else(|_| if vulkan { "2" } else { "8" }.to_string())
}

fn main() {
    write_sizes();
    // `--features vulkan` (`Cargo.toml`), which Cargo hands to build scripts as
    // `CARGO_FEATURE_VULKAN`.
    let vulkan = std::env::var_os("CARGO_FEATURE_VULKAN").is_some();
    let mut build = cleave_build::Build::new();
    if small() {
        for (name, value) in SMALL {
            build.define(*name, *value);
        }
    }
    build.define("SPLIT", split(vulkan));
    build
        .file("src/kernel.cleave")
        .opt_level(OPT_LEVEL)
        .openmp(OPENMP)
        .inline(INLINE)
        .debug_info(DEBUG)
        .llvm_loop_unroll(LLVM_LOOP_UNROLL)
        .target_cpu("native")
        .backend(if vulkan { cleave_build::Backend::Vulkan } else { cleave_build::Backend::Cpu })
        .compile("kernel");
}
