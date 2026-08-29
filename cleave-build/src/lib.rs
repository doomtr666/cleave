//! Compile-time helper for a `build.rs` that wants to turn one or more
//! `.cleave` source files into a real, linkable object file plus generated
//! Rust FFI bindings for every `export fn` in them -- the same shape
//! `ispc_compile`/`ispc-rs` gives ISPC, but calling straight into
//! `cleave::pipeline` *in process* rather than shelling out to a separately
//! installed compiler binary (`cleave` already *is* a Rust library, unlike
//! ISPC, so there's nothing to subprocess or find on `PATH`).
//!
//! ```no_run
//! // build.rs
//! fn main() {
//!     cleave_build::compile_library("kernel", &["src/kernel.cleave"]);
//! }
//! ```
//!
//! This links the emitted object directly (`cargo:rustc-link-arg`) rather
//! than archiving it into a `.lib`/`.a` first (`ispc-rs`'s own approach,
//! needed there because it compiles one object per `.ispc` file and has to
//! combine several) -- fine for now since `compile`/`compile_library` each
//! take one whole source set and produce exactly one object; revisit if a
//! real use ever needs several independently-cached objects linked
//! together.
//!
//! Deliberately not published/generalized beyond this workspace yet:
//! `cleave-rt` is located as a fixed sibling directory of `cleave-build`
//! itself (`env!("CARGO_MANIFEST_DIR")/../cleave-rt` -- compile-time, *this*
//! crate's own directory, not whichever consumer's build script ends up
//! running this code -- see `build_cleave_rt`'s own doc comment) -- correct
//! for *this* workspace's own fixed layout, not a consumer outside it.
//!
//! `cleave-rt`'s own staticlib is built by this module directly (a nested
//! `cargo build` invocation, into its own dedicated target directory under
//! `OUT_DIR`) rather than assumed to already exist -- found by direct
//! testing that it's genuinely necessary: Cargo does *not* materialize a
//! dependency's extra declared `crate-type`s (`cleave-rt/Cargo.toml`'s own
//! `["rlib", "staticlib"]`) just because it's pulled in transitively (via
//! `cleave` here) -- only an ordinary `rlib` gets built and linked into
//! `cleave` itself. Only `cargo build -p cleave-rt` (or `--workspace`),
//! naming it as a build *target* in its own right, produces the `.lib`/
//! `.a`. An earlier version of this module assumed the artifact would
//! simply already be there (reasoning: "Cargo must have built cleave-rt
//! already to build `cleave`/`cleave-build` itself") -- true for the
//! `.rlib`, but not the `.lib`, and this went unnoticed until a `--release`
//! build (never separately `cargo build -p cleave-rt --release`d by hand
//! the way `--debug` happened to have been) hit `LNK1181: cannot open
//! input file 'cleave_rt.lib'`. A dedicated target directory (rather than
//! reusing the outer build's own shared one) sidesteps any risk of lock
//! contention between this nested `cargo build` and the outer one already
//! in progress -- the cost is `cleave-rt` (small, fast to compile, no heavy
//! dependencies) rebuilding once per consuming crate rather than being
//! shared, an acceptable trade for correctness here.

use std::env;
use std::path::{Path, PathBuf};

/// Convenience wrapper matching `ispc::compile_library`'s own shape:
/// `cleave_build::compile_library("kernel", &["src/kernel.cleave"])`.
pub fn compile_library(name: &str, files: &[&str]) {
    let mut build = Build::new();
    for f in files {
        build.file(f);
    }
    build.compile(name);
}

pub struct Build {
    files: Vec<PathBuf>,
}

impl Default for Build {
    fn default() -> Self {
        Self::new()
    }
}

impl Build {
    pub fn new() -> Self {
        Build { files: Vec::new() }
    }

    pub fn file(&mut self, path: impl Into<PathBuf>) -> &mut Self {
        self.files.push(path.into());
        self
    }

    /// Compiles every file added via `file` as one merged program (`use`
    /// resolution rooted at each file's own parent directory, the shipped
    /// stdlib always available as a fallback -- exactly `cleave`'s own CLI
    /// convention, `main.rs`'s own `real_main`), and emits, into `OUT_DIR`:
    /// `<name>.o` (linked in directly), `<name>_bindings.rs` (meant to be
    /// `include!`'d by the consuming crate's own source).
    ///
    /// Panics on any compile/type/link-setup error, with every collected
    /// message included -- the conventional way a build-script helper
    /// reports failure (mirrors `cc::Build::compile`'s own posture); Cargo
    /// surfaces a panicking build script's own output directly.
    pub fn compile(&self, name: &str) {
        assert!(!self.files.is_empty(), "cleave_build::Build::compile(\"{name}\"): no source files added via `.file(...)`");

        for f in &self.files {
            println!("cargo:rerun-if-changed={}", f.display());
        }

        let out_dir = PathBuf::from(env::var("OUT_DIR").expect("cleave-build must run inside a build script (OUT_DIR unset)"));
        let object_path = out_dir.join(format!("{name}.o"));
        let bindings_path = out_dir.join(format!("{name}_bindings.rs"));

        let mut sources = Vec::with_capacity(self.files.len());
        let mut project_dirs: Vec<PathBuf> = Vec::new();
        for f in &self.files {
            let text = std::fs::read_to_string(f).unwrap_or_else(|e| panic!("cleave-build: failed to read {}: {e}", f.display()));
            sources.push((f.display().to_string(), text));
            if let Some(dir) = f.parent() {
                if !project_dirs.contains(&dir.to_path_buf()) {
                    project_dirs.push(dir.to_path_buf());
                }
            }
        }

        // Run on a dedicated, generously-sized stack -- `doc/backlog.md`'s
        // own "Multi-level call transparency in the e-graph forward
        // translator" item: `derive()`'s own recursive e-graph exploration
        // can recurse deep enough to overflow the *default* 2MB stack a
        // thread gets (confirmed there for `cargo test`'s own worker
        // threads, fixed on that side by setting `RUST_MIN_STACK=64MiB`
        // before running). A build script has no equivalent escape hatch of
        // its own to reach for -- it just *is* whatever thread Cargo spawns
        // it on, with whatever default stack that thread got, no env var a
        // build-script author can set from inside their own `build.rs` to
        // change it after the fact. Found for real, not hypothetical:
        // `examples/digits-interop`'s own kernel (a real, wider-than-`xor`
        // MLP -- `doc/backlog.md`'s own "Char literals"/`println` items'
        // neighbor, "stress the system to find bugs" in practice) hit this
        // exact crash (`STATUS_STACK_OVERFLOW`) the first time anything
        // bigger than `xor_tensor.cleave`'s own 2-neuron network was ever
        // compiled through `cleave-build` specifically. `std::thread::scope`
        // (stable since 1.63) lets the spawned thread still borrow `sources`
        // /`project_dirs`/the two `Path`s directly, no cloning into `'static`
        // needed just to give it its own stack.
        let result = std::thread::scope(|scope| {
            std::thread::Builder::new()
                .stack_size(1024 * 1024 * 1024)
                .spawn_scoped(scope, || {
                    cleave::pipeline::compile_and_emit(
                        sources,
                        &project_dirs,
                        Some(&object_path),
                        Some(&bindings_path),
                    )
                })
                .expect("cleave-build: failed to spawn the compile thread")
                .join()
                .unwrap_or_else(|e| std::panic::resume_unwind(e))
        });
        if let Err(errs) = result {
            panic!("cleave-build: failed to compile `{name}`:\n{}", errs.join("\n"));
        }

        println!("cargo:rustc-link-arg={}", object_path.display());
        println!("cargo:rustc-link-lib=static=cleave_rt");
        println!("cargo:rustc-link-search=native={}", build_cleave_rt(&out_dir).display());

        // `libomp` -- needed the moment `emit_object`'s own OpenMP
        // parallelization stage is ever exercised (`cleave::pipeline::
        // register_openmp_stub_symbols`'s own doc comment: the emitted
        // object always carries real, unresolved `__kmpc_*` relocations
        // once any linalg-derived kernel exists at all, unconditionally --
        // confirmed directly, the first `cargo build` of a real interop
        // example through this exact function after that stage landed
        // failed with `LNK2019: symbole externe non résolu __kmpc_*`). The
        // real, installed file is named `libomp.lib` (LLVM's own cross-
        // platform convention) -- `dylib=libomp`, not `dylib=omp`, since an
        // `-msvc` target's linker takes an `-l`-equivalent name verbatim
        // (`NAME.lib`), no GNU-style prefix assumed. Same `MLIR_SYS_220_
        // PREFIX` (`.cargo/config.toml`) `mlir-sys`'s own build script
        // already keys off of, reused rather than a second, independently-
        // maintained path -- cargo's own `[env]` mechanism hands it to
        // every build-script process, this one included.
        let mlir_prefix = env::var("MLIR_SYS_220_PREFIX")
            .expect("MLIR_SYS_220_PREFIX must be set (see .cargo/config.toml) to link libomp");
        println!("cargo:rustc-link-lib=dylib=libomp");
        println!("cargo:rustc-link-search=native={mlir_prefix}/lib");

        // `libomp.dll` itself, copied next to the *final* binary -- linking
        // above only satisfies the linker; `libomp` is a real, dynamically-
        // loaded runtime (Windows has no static-link story for it), so the
        // finished executable also needs the `.dll` findable through its
        // own DLL search order at process-start time, same as any other
        // runtime dependency Windows resolves outside of `PATH`/`rustc`'s
        // own knowledge. `OUT_DIR` (`target/<profile>/build/<pkg-hash>/
        // out`) sits exactly three directories below `target/<profile>/`,
        // where cargo actually places the finished binary -- confirmed
        // directly against a real build, not assumed. Best-effort
        // (`let _ =`, not `.expect(...)`): a failed copy here shouldn't
        // fail the whole build over what's fundamentally a convenience for
        // running the binary straight out of `target/`, and every other
        // real deployment path (an installed/packaged binary) needs its
        // own real answer to "where does `libomp.dll` come from" anyway,
        // not this one.
        if let Some(target_profile_dir) = out_dir.ancestors().nth(3) {
            let _ = std::fs::copy(
                format!("{mlir_prefix}/bin/libomp.dll"),
                target_profile_dir.join("libomp.dll"),
            );
        }
    }
}

/// Builds `cleave-rt`'s own staticlib into `<out_dir>/cleave-rt-target/
/// <profile>/` (a dedicated target directory, not the outer build's shared
/// one -- see the module's own doc comment for why) and returns that
/// directory, ready for `cargo:rustc-link-search=native=...`.
fn build_cleave_rt(out_dir: &Path) -> PathBuf {
    // `env!` (compile-time), not `env::var` (runtime): this code runs
    // *inside* the consuming crate's own build script process, where a
    // runtime `CARGO_MANIFEST_DIR` lookup would give *that* crate's own
    // directory (found by direct testing: `rust-interop-demo`'s, producing
    // a nonsensical `.../rust-interop-demo/../cleave-rt` path that doesn't
    // exist) -- `env!` instead bakes in *this* crate's (`cleave-build`'s)
    // own manifest directory at the point `cleave-build` itself was
    // compiled, which is what "a fixed sibling in this workspace's own
    // layout" actually needs to be relative to.
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let cleave_rt_manifest = manifest_dir.join("../cleave-rt/Cargo.toml");

    // `PROFILE` (a build-script env var Cargo always sets) is exactly
    // "debug" or "release" -- conveniently, the same two strings Cargo's
    // own target-directory naming uses, so no separate translation needed
    // between "which profile is the outer build using" and "which
    // subdirectory will the nested build's own output land in".
    let profile = env::var("PROFILE").unwrap_or_else(|_| "debug".to_string());
    let target_dir = out_dir.join("cleave-rt-target");

    let cargo = env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    let mut cmd = std::process::Command::new(cargo);
    cmd.arg("build").arg("--manifest-path").arg(&cleave_rt_manifest).arg("--target-dir").arg(&target_dir).arg("--quiet");
    if profile == "release" {
        cmd.arg("--release");
    }
    let status = cmd
        .status()
        .unwrap_or_else(|e| panic!("cleave-build: failed to run `cargo build` for cleave-rt's own staticlib: {e}"));
    assert!(status.success(), "cleave-build: `cargo build` for cleave-rt's own staticlib failed (exit status: {status})");

    target_dir.join(profile)
}
