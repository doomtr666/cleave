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
//! Deliberately not published/generalized beyond this workspace yet: the
//! `cleave-rt` staticlib is located by walking up from `OUT_DIR` to the
//! shared `target/<profile>/` directory (see `runtime_search_dir` below) --
//! correct for an ordinary, non-cross-compiled build within *this*
//! workspace, not yet handling a `--target <triple>` build (which inserts
//! an extra `target/<triple>/` path segment) or a consumer outside this
//! workspace's own `cleave-rt`.

use std::env;
use std::path::PathBuf;

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

        if let Err(errs) = cleave::pipeline::compile_and_emit(sources, &project_dirs, Some(&object_path), Some(&bindings_path)) {
            panic!("cleave-build: failed to compile `{name}`:\n{}", errs.join("\n"));
        }

        println!("cargo:rustc-link-arg={}", object_path.display());
        println!("cargo:rustc-link-lib=static=cleave_rt");
        println!("cargo:rustc-link-search=native={}", runtime_search_dir().display());
    }
}

/// `OUT_DIR` for a build script is `target/<profile>/build/<pkg>-<hash>/out`
/// -- three levels up is the shared `target/<profile>/` directory every
/// crate-type artifact in this workspace's build lands in, including
/// `cleave-rt`'s own `cleave_rt.lib`/`.a` (`cleave-rt/Cargo.toml`'s own
/// `crate-type = ["rlib", "staticlib"]`). Reliable *because* `cleave-rt` is
/// a transitive dependency of `cleave-build` itself (via `cleave`) -- Cargo
/// must already have built it before a consumer's `build.rs` (which itself
/// depends on `cleave-build`) can even run.
fn runtime_search_dir() -> PathBuf {
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("cleave-build must run inside a build script (OUT_DIR unset)"));
    out_dir
        .ancestors()
        .nth(3)
        .unwrap_or_else(|| panic!("cleave-build: OUT_DIR {} has an unexpected shape", out_dir.display()))
        .to_path_buf()
}
