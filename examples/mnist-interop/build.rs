// Compiles `src/kernel.cleave` -- see `examples/rust-interop-demo/build.rs`'s
// own identical comment for the mechanism.
fn main() {
    cleave_build::compile_library("kernel", &["src/kernel.cleave"]);
}
