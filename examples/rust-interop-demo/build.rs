// Compiles `src/kernel.cleave` in process (via `cleave-build`, itself
// calling straight into `cleave::pipeline` -- no external `cleave` binary
// on `PATH` needed) into a real object file, linked directly into this
// crate, plus `kernel_bindings.rs` in `OUT_DIR`, `include!`'d by `main.rs`.
//
// Linker note: this links via whatever linker Cargo/rustc already default
// to on this platform (MSVC's `link.exe` on Windows) -- fine for proving
// the mechanism. To link via the LLVM toolchain instead (`clang`/`lld-
// link`, no MSVC involved at all -- `cleave` itself already depends on
// LLVM/MLIR, so this is a natural alternative), point `RUSTFLAGS` or a
// local `.cargo/config.toml`'s `[target.<triple>] linker = "..."` at a real
// `clang`/`lld-link` on your own machine; not hardcoded here since that
// path is machine-specific.
fn main() {
    cleave_build::compile_library("kernel", &["src/kernel.cleave"]);
}
