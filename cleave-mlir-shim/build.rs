use std::env;

fn main() {
    println!("cargo:rerun-if-changed=cpp/shim.cpp");
    println!("cargo:rerun-if-env-changed=MLIR_SYS_220_PREFIX");

    // Same env var `mlir-sys` itself reads (`mlir-sys-220.0.2/build.rs`) --
    // temporary until `cleave-llvm-redist`'s own prebuilt release exists;
    // see this crate's own `Cargo.toml` doc comment.
    let prefix = env::var("MLIR_SYS_220_PREFIX")
        .expect("MLIR_SYS_220_PREFIX must be set (see .cargo/config.toml) to build cleave-mlir-shim");

    cc::Build::new()
        .cpp(true)
        .std("c++17")
        // Matches the exact settings the vendored LLVM/MLIR toolchain was
        // itself built with (`LLVM_ENABLE_RTTI=OFF`/`LLVM_ENABLE_EH=OFF`,
        // confirmed directly against `I:/Dev/llvm-project/build/
        // CMakeCache.txt`, not assumed) -- mismatching either here risks a
        // real ABI mismatch across the boundary between this shim's own
        // object file and the prebuilt `libMLIR*`/`LLVM*` static libs it
        // links against.
        .flag_if_supported("/GR-") // MSVC: disable RTTI
        .flag_if_supported("-fno-rtti") // clang-cl: disable RTTI
        .flag_if_supported("/EHs-c-") // MSVC: disable exceptions
        .flag_if_supported("-fno-exceptions") // clang-cl: disable exceptions
        .include(format!("{prefix}/include"))
        .file("cpp/shim.cpp")
        .compile("cleave_mlir_shim");
}
