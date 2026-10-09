use std::env;

fn main() {
    println!("cargo:rerun-if-changed=cpp/shim.cpp");
    println!("cargo:rerun-if-env-changed=CLEAVE_LLVM_PREFIX");

    // The LLVM/MLIR install: `scripts/setup-toolchain.ps1` fetches the
    // prebuilt one (`cleave-llvm-redist`) and writes this variable into
    // `.cargo/config.toml`.
    let prefix = env::var("CLEAVE_LLVM_PREFIX")
        .expect("CLEAVE_LLVM_PREFIX must be set (see .cargo/config.toml) to build cleave-mlir-shim");

    cc::Build::new()
        .cpp(true)
        .std("c++17")
        // Always the release CRT/STL (`/MD`, no `_DEBUG`, `_ITERATOR_DEBUG_
        // LEVEL=0`), whatever cargo profile is building this crate: the
        // vendored `libMLIR*`/`LLVM*` are release builds, and this shim
        // hands them STL objects (`ExecutionEngineOptions::transformer` is
        // a `std::function`). Under a debug cargo profile `cc-rs` would
        // default to `/MDd`, giving those objects a different layout than
        // the release code calling them expects -- found as a hard
        // `STATUS_STACK_BUFFER_OVERRUN` in every `dev`-profile build (build
        // scripts included) the moment the shim built its own
        // `std::function` wrapper instead of just copying upstream's.
        .opt_level(2)
        .debug(false)
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

    link_llvm_and_mlir(&prefix);
}

/// Links the shim's dependencies, statically: every MLIR library in the
/// prefix (MLIR's C API among them, which `llvm-config` doesn't list), every
/// LLVM component `llvm-config --libnames` lists, and the system libraries
/// they need.
fn link_llvm_and_mlir(prefix: &str) {
    let lib_dir = format!("{prefix}/lib");
    println!("cargo:rustc-link-search=native={lib_dir}");
    let static_name = |file: &str| -> Option<String> {
        file.strip_suffix(".lib")
            .or_else(|| file.strip_prefix("lib").and_then(|f| f.strip_suffix(".a")))
            .map(str::to_string)
    };
    let mut mlir: Vec<String> = std::fs::read_dir(&lib_dir)
        .unwrap_or_else(|e| panic!("cannot read {lib_dir}: {e}"))
        .filter_map(|entry| entry.ok()?.file_name().into_string().ok())
        .filter(|file| file.starts_with("MLIR") || file.starts_with("libMLIR"))
        .filter_map(|file| static_name(&file))
        .collect();
    mlir.sort();
    for name in mlir {
        println!("cargo:rustc-link-lib=static={name}");
    }
    let llvm_config = |argument: &str| -> String {
        let tool = format!("{prefix}/bin/llvm-config{}", std::env::consts::EXE_SUFFIX);
        let output = std::process::Command::new(&tool)
            .args([argument, "--link-static"])
            .output()
            .unwrap_or_else(|e| panic!("cannot run {tool}: {e}"));
        assert!(output.status.success(), "{tool} {argument} failed");
        String::from_utf8_lossy(&output.stdout).trim().to_string()
    };
    for file in llvm_config("--libnames").split_whitespace() {
        if let Some(name) = static_name(file) {
            println!("cargo:rustc-link-lib=static={name}");
        }
    }
    for flag in llvm_config("--system-libs").split_whitespace() {
        let flag = flag.trim_start_matches("-l");
        let flag = flag.strip_suffix(".lib").unwrap_or(flag);
        if !flag.is_empty() {
            println!("cargo:rustc-link-lib={flag}");
        }
    }
}
