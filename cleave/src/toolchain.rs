//! Where the LLVM/MLIR toolchain cleave builds against is, and what of it a
//! compiled program needs at run time: the OpenMP runtime (`libomp`), for
//! `--openmp`'s parallel loops and `spawn`'s tasks. The platform differences
//! live here, once.

/// The toolchain's install prefix (`CLEAVE_LLVM_PREFIX`, written into
/// `.cargo/config.toml` by `scripts/setup-toolchain.ps1`).
pub fn llvm_prefix() -> Result<String, String> {
    std::env::var("CLEAVE_LLVM_PREFIX")
        .map_err(|_| "CLEAVE_LLVM_PREFIX must be set (see .cargo/config.toml)".to_string())
}

/// `libomp`'s name for the linker (`-l <name>`): an MSVC linker takes it
/// verbatim (`libomp.lib`), a GNU-style one adds the `lib` prefix itself.
pub const LIBOMP_LINK_NAME: &str = if cfg!(target_env = "msvc") { "libomp" } else { "omp" };

/// The directory holding `libomp`'s import or shared library, to link against.
pub fn libomp_link_dir(prefix: &str) -> String {
    format!("{prefix}/lib")
}

/// `libomp`'s shared library, which a running program loads: a DLL in `bin`
/// on Windows, a shared object in `lib` elsewhere.
pub fn libomp_shared_library(prefix: &str) -> String {
    if cfg!(windows) {
        format!("{prefix}/bin/libomp.dll")
    } else if cfg!(target_os = "macos") {
        format!("{prefix}/lib/libomp.dylib")
    } else {
        format!("{prefix}/lib/libomp.so")
    }
}
