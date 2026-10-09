use std::env;

fn main() {
    println!("cargo:rerun-if-changed=cpp/shim.cpp");
    println!("cargo:rerun-if-env-changed=CLEAVE_LLVM_PREFIX");

    // The LLVM/MLIR install: `scripts/setup-toolchain.ps1` fetches the
    // prebuilt one (`cleave-llvm-redist`) and writes this variable into
    // `.cargo/config.toml`.
    let prefix = env::var("CLEAVE_LLVM_PREFIX")
        .expect("CLEAVE_LLVM_PREFIX must be set (see .cargo/config.toml) to build cleave-mlir");

    cc::Build::new()
        .cpp(true)
        .std("c++17")
        // Built the way the toolchain was, whatever the cargo profile, since
        // the shim hands the toolchain's libraries STL objects (a
        // `std::function` among them): the release runtime (under a debug
        // profile `cc` would pick MSVC's debug CRT, a different layout, a
        // crash), no RTTI, no exceptions (`LLVM_ENABLE_RTTI=OFF`,
        // `LLVM_ENABLE_EH=OFF`). Each flag in its MSVC and GCC/Clang spelling.
        .opt_level(2)
        .debug(false)
        .flag_if_supported("/GR-")
        .flag_if_supported("-fno-rtti")
        .flag_if_supported("/EHs-c-")
        .flag_if_supported("-fno-exceptions")
        .include(format!("{prefix}/include"))
        .file("cpp/shim.cpp")
        .compile("cleave_mlir");

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
