//! A single, real, thread-local `CodegenOptions` context, read by every
//! deeply-nested pass that used to read a scattered `CLEAVE_*` env var
//! instead (`unroll_jam.rs`/`chain_split.rs`/`mlir_lower.rs`/`dps_rewrite.rs`
//! -- see `doc/backlog.md`'s own entry on this). Raised directly by the
//! user, unhappy with env vars specifically for being invisible and easy to
//! lose track of ("on ne sait jamais où on en est") -- the fix isn't to
//! delete the env vars and thread a `&CodegenOptions` parameter through
//! every intermediate function instead: `lower_program` alone has 18 call
//! sites (4 real, 14 test files), and several of the gated passes sit many
//! layers below their own nearest `CodegenOptions`-aware caller. A
//! `thread_local!` context, set once at the top (`main.rs`'s own CLI
//! dispatch, `cleave-build::Build::compile`) and read via `current()`
//! anywhere below, gives the same real, typed, CLI-controllable behavior
//! (no more stringly-typed, easy-to-forget env vars) without changing any
//! of those 18 signatures.
//!
//! **Why `thread_local!`, not a plain process-global `static`/`OnceCell`**:
//! `cargo test`'s own default harness runs different `#[test]` functions
//! concurrently, each potentially wanting *different* options -- a shared
//! process-global would let one test's own settings leak into another
//! running at the same time. A `thread_local!` doesn't fully solve this
//! either (the test harness reuses a fixed-size worker-thread pool, not one
//! OS thread per test, so a later test scheduled onto the same worker
//! thread as an earlier one *can* still observe that earlier test's own
//! last-set options if it doesn't set its own) -- but this is strictly
//! better than a process-global env var (today's actual behavior, shared
//! across literally every thread) and cheap to make fully safe in practice:
//! any test that cares about non-default options must call `set` at its own
//! start, never assume "probably still default" -- exactly the discipline a
//! good test should already follow regardless of this module's own
//! isolation guarantees.

use crate::pipeline::CodegenOptions;
use std::cell::RefCell;

thread_local! {
    static CURRENT: RefCell<CodegenOptions> = RefCell::new(CodegenOptions::default());
}

/// Replaces this thread's own current options wholesale. Call once, before
/// any compilation work that consults `current()` below -- `main.rs`'s own
/// CLI dispatch (right after resolving `CodegenOptions` from parsed flags)
/// and `cleave-build::Build::compile` (right after resolving its own
/// builder-configured options) are the two real call sites.
pub fn set(options: CodegenOptions) {
    CURRENT.with(|cell| *cell.borrow_mut() = options);
}

/// This thread's own current options -- `CodegenOptions::default()` if
/// `set` was never called on it. Cheap to call freely (a plain `Clone` of a
/// small struct), not meant to be cached across a call that might itself
/// call `set` again.
pub fn current() -> CodegenOptions {
    CURRENT.with(|cell| cell.borrow().clone())
}
