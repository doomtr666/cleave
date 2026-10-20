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

/// The directory holding `libomp`'s import or shared library, to link against:
/// `lib` on Windows; on Linux, `lib/<target triple>`, where LLVM installs its
/// runtimes there (`LLVM_ENABLE_PER_TARGET_RUNTIME_DIR`, on by default: the
/// `cleave-llvm-redist` Linux toolchain's `libomp.so` is in
/// `lib/x86_64-unknown-linux-gnu`), else `lib`.
pub fn libomp_link_dir(prefix: &str) -> String {
    if cfg!(target_os = "linux") {
        let per_target = format!("{prefix}/lib/{}-unknown-linux-gnu", std::env::consts::ARCH);
        if std::path::Path::new(&per_target).join("libomp.so").exists() {
            return per_target;
        }
    }
    format!("{prefix}/lib")
}

/// `libomp`'s shared library, which a running program loads: a DLL in `bin`
/// on Windows, a shared object in its link directory elsewhere.
pub fn libomp_shared_library(prefix: &str) -> String {
    if cfg!(windows) {
        format!("{prefix}/bin/libomp.dll")
    } else if cfg!(target_os = "macos") {
        format!("{}/libomp.dylib", libomp_link_dir(prefix))
    } else {
        format!("{}/libomp.so", libomp_link_dir(prefix))
    }
}
