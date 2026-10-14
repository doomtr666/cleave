use std::env;

/// The C++ sources, compiled each to an object of its own: an edit to one
/// recompiles that one only (`cached_object`).
const SOURCES: [&str; 2] = ["cpp/shim.cpp", "cpp/gpu.cpp"];
/// What every source includes first.
const HEADER: &str = "cpp/cleave.h";
/// Compiled alone, with MSVC, to precompile `HEADER`.
const PCH_SOURCE: &str = "cpp/pch.cpp";
/// Data a source includes (the tuned products' table), part of its
/// objects' hash.
const INCLUDED: [&str; 1] = ["cpp/matmul_table.inc"];

fn main() {
    for file in SOURCES.iter().chain([&HEADER, &PCH_SOURCE]).chain(INCLUDED.iter()) {
        println!("cargo:rerun-if-changed={file}");
    }
    println!("cargo:rerun-if-env-changed=CLEAVE_LLVM_PREFIX");

    // The LLVM/MLIR install: `scripts/setup-toolchain.ps1` fetches the
    // prebuilt one (`cleave-llvm-redist`) and writes this variable into
    // `.cargo/config.toml`.
    let prefix = env::var("CLEAVE_LLVM_PREFIX")
        .expect("CLEAVE_LLVM_PREFIX must be set (see .cargo/config.toml) to build cleave-mlir");

    // With MSVC, the header precompiled once (its object linked too): it is
    // most of a source's compile time, MLIR's headers.
    let pch = build(&prefix).get_compiler().is_like_msvc().then(|| precompiled_header(&prefix));
    // The sources whose object isn't cached compile in parallel, each in a
    // thread of its own.
    let mut objects: Vec<std::path::PathBuf> = std::thread::scope(|scope| {
        let builds: Vec<_> = SOURCES
            .iter()
            .map(|source| scope.spawn(|| cached_object(source, &prefix, pch.as_ref().map(|(pch, _)| pch.as_path()))))
            .collect();
        builds.into_iter().map(|build| build.join().expect("compiling the shim")).collect()
    });
    if let Some((_, object)) = pch {
        objects.push(object);
    }
    let mut archive = cc::Build::new();
    archive.cpp(true);
    for object in &objects {
        archive.object(object);
    }
    archive.compile("cleave_mlir");

    link_llvm_and_mlir(&prefix);
}

/// The compiler settings every source is built with.
fn build(prefix: &str) -> cc::Build {
    let mut build = cc::Build::new();
    build.cpp(true).std("c++17").opt_level(2).debug(false).include(format!("{prefix}/include"));
    // Built the way the toolchain was, whatever the cargo profile, since the
    // shim hands the toolchain's libraries STL objects (a `std::function`
    // among them): the release runtime (under a debug profile `cc` would pick
    // MSVC's debug CRT, a different layout, a crash), no RTTI, no exceptions
    // (`LLVM_ENABLE_RTTI=OFF`, `LLVM_ENABLE_EH=OFF`). Each flag in its
    // compiler's spelling, chosen rather than probed (`flag_if_supported`
    // runs the compiler once per flag, per build).
    if build.get_compiler().is_like_msvc() {
        build.flag("/GR-").flag("/EHs-c-");
    } else {
        build.flag("-fno-rtti").flag("-fno-exceptions");
    }
    build
}

/// A hash of `files` and `prefix`, naming what's compiled from them.
fn content_hash(files: &[&str], prefix: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for file in files {
        std::fs::read(file).unwrap_or_else(|e| panic!("reading {file}: {e}")).hash(&mut hasher);
    }
    prefix.hash(&mut hasher);
    hasher.finish()
}

fn out_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"))
}

/// Removes what earlier builds cached under `prefix` (`gpu-`, `cleave-`)
/// with the extension `extension`, but `current`: a precompiled header is
/// over a gigabyte.
fn prune(prefix: &str, extension: &str, current: &std::path::Path) {
    let Ok(entries) = std::fs::read_dir(out_dir()) else { return };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().into_owned();
        if path != current && name.starts_with(prefix) && name.ends_with(extension) {
            let _ = std::fs::remove_file(path);
        }
    }
}

/// MSVC's precompiled `HEADER` and the object that comes with it, made
/// unless already in `OUT_DIR` for this header.
fn precompiled_header(prefix: &str) -> (std::path::PathBuf, std::path::PathBuf) {
    let hash = content_hash(&[HEADER, PCH_SOURCE], prefix);
    let pch = out_dir().join(format!("cleave-{hash:016x}.pch"));
    let object = out_dir().join(format!("pch-{hash:016x}.o"));
    if !pch.exists() || !object.exists() {
        let compiled = build(prefix)
            .flag("/Yccleave.h")
            .flag(format!("/Fp{}", pch.display()))
            .file(PCH_SOURCE)
            .compile_intermediates();
        std::fs::copy(&compiled[0], &object).expect("caching the header's object");
    }
    prune("cleave-", ".pch", &pch);
    prune("pch-", ".o", &object);
    (pch, object)
}

/// `source`'s object, compiled unless one of the same source, header and
/// prefix is already in `OUT_DIR`: `cc` recompiles every file each time the
/// build script runs, and a source is seconds of MLIR headers even
/// precompiled.
fn cached_object(source: &str, prefix: &str, pch: Option<&std::path::Path>) -> std::path::PathBuf {
    let mut hashed = vec![source, HEADER];
    hashed.extend(INCLUDED);
    let hash = content_hash(&hashed, prefix);
    let stem = std::path::Path::new(source).file_stem().and_then(|s| s.to_str()).expect("a source name");
    let cached = out_dir().join(format!("{stem}-{hash:016x}.o"));
    if !cached.exists() {
        let mut build = build(prefix);
        if let Some(pch) = pch {
            build.flag("/Yucleave.h").flag(format!("/Fp{}", pch.display()));
        }
        let compiled = build.file(source).compile_intermediates();
        std::fs::copy(&compiled[0], &cached).expect("caching an object");
    }
    prune(&format!("{stem}-"), ".o", &cached);
    cached
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
