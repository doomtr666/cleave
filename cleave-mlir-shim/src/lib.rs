//! The first real cleave shim: one function, `cleaveExecutionEngineCreate
//! WithTarget` (`cpp/shim.cpp`), giving real `--target-cpu`/`--target-
//! features` control over the `TargetMachine` `mlir::ExecutionEngine`
//! actually JIT-compiles with -- a real gap `melior`/`mlir-sys` can't close
//! themselves (`doc/backlog.md`'s own entry has the full story: `mlirExecu
//! tionEngineCreate`'s C API always calls `JITTargetMachineBuilder::detect
//! Host()` with no override hook at all).
//!
//! Deliberately minimal, matching this project's own recommended sequencing
//! for the shim as a whole (`doc/backlog.md`'s melior/mlir-sys decision
//! entry): one function, no new types beyond a thin wrapper mirroring
//! `melior::ExecutionEngine`'s own shape, depends on `mlir-sys` directly
//! (not `melior`, whose own `ExecutionEngine` has a private `raw` field
//! this crate has no way to construct from an externally-built handle) --
//! proves the build/link/FFI shape on the narrowest possible real surface
//! before deciding whether to extend it.

use mlir_sys::{
    MlirExecutionEngine, MlirModule, MlirOperation, MlirStringRef, mlirExecutionEngineDestroy,
    mlirExecutionEngineDumpToObjectFile, mlirExecutionEngineInvokePacked,
    mlirExecutionEngineLookup, mlirExecutionEngineRegisterSymbol,
};

unsafe extern "C" {
    fn cleaveExecutionEngineCreateWithTarget(
        op: MlirModule,
        opt_level: i32,
        num_paths: i32,
        shared_lib_paths: *const MlirStringRef,
        enable_object_dump: bool,
        enable_pic: bool,
        target_cpu: MlirStringRef,
        target_features: MlirStringRef,
        loop_unroll: bool,
    ) -> MlirExecutionEngine;
    fn cleaveApproximateMath(op: MlirOperation) -> bool;
    fn cleaveHoistArgSlots(op: MlirOperation);
    fn cleaveLowerAdoptions(op: MlirOperation);
    fn cleaveLimitInlining(op: MlirOperation, threshold: i64) -> i64;
    fn cleaveApplyNoInline(op: MlirOperation);
    fn cleaveCopyAggregatesInMemory(op: MlirOperation, min_bytes: i64) -> i64;
    fn cleaveLowerSpawns(op: MlirOperation, tasks: bool) -> bool;
}

/// Turns `spawn`'s markers into OpenMP tasks after bufferization
/// (`cpp/shim.cpp`'s `cleaveLowerSpawns`): each spawned call in an
/// `omp.task`, each wait an `omp.taskwait`, each spawning function wrapped to
/// run on a parallel region's team. With `tasks` false, the markers are only
/// removed: each spawned call runs in place, no wait, no OpenMP (serial
/// elision). `false` if `op` isn't a module.
///
/// # Safety
///
/// `op` must be a valid module, not used concurrently.
pub unsafe fn lower_spawns(op: MlirOperation, tasks: bool) -> bool {
    unsafe { cleaveLowerSpawns(op, tasks) }
}

/// Rewrites `tanh`/`exp`/`log` and their relatives under `op` into
/// polynomial approximations (`cpp/shim.cpp`'s `cleaveApproximateMath`), so
/// they vectorize instead of becoming one libm call per vector element.
/// `false` if the rewrite didn't converge.
///
/// # Safety
///
/// `op` must be a valid operation, not used concurrently.
pub unsafe fn approximate_math(op: MlirOperation) -> bool {
    unsafe { cleaveApproximateMath(op) }
}

/// Moves every argument slot (`cleave.arg_slot`, `cleave.spawn_arg_slot`)
/// an inlined call left in a non-entry block of its function back to the
/// entry block, and bounds an ordinary call's slot's lifetime to its uses so
/// that LLVM can share storage between slots (`cpp/shim.cpp`'s
/// `cleaveHoistArgSlots`). Run once loops are blocks.
///
/// # Safety
///
/// `op` must be a valid operation, not used concurrently.
pub unsafe fn hoist_arg_slots(op: MlirOperation) {
    unsafe { cleaveHoistArgSlots(op) }
}

/// Turns each adoption (a `bufferization.clone` marked `cleave.adopt`,
/// `mlir_lower.rs`'s `PrimOp::Adopt`) into a retain of the same buffer
/// (`cpp/shim.cpp`'s `cleaveLowerAdoptions`). Run after the buffer
/// deallocation passes, before `bufferization-to-memref`.
///
/// # Safety
///
/// `op` must be a valid `builtin.module`, not used concurrently.
pub unsafe fn lower_adoptions(op: MlirOperation) {
    unsafe { cleaveLowerAdoptions(op) }
}

/// Marks `no_inline` every `func.call` whose callee, inlined, would exceed
/// `threshold` operations: its own body plus everything it would inline in
/// turn (`cpp/shim.cpp`'s `cleaveLimitInlining`). Returns how many calls it
/// marked. Run right before MLIR's inliner.
///
/// # Safety
///
/// `op` must be a valid `builtin.module`, not used concurrently.
pub unsafe fn limit_inlining(op: MlirOperation, threshold: i64) -> i64 {
    unsafe { cleaveLimitInlining(op, threshold) }
}

/// Gives LLVM's `noinline` to every `llvm.func` [`limit_inlining`] marked
/// (`cpp/shim.cpp`'s `cleaveApplyNoInline`), so LLVM's own inliner keeps it
/// out of line too. Run once the module is in the LLVM dialect.
///
/// # Safety
///
/// `op` must be a valid operation, not used concurrently.
pub unsafe fn apply_no_inline(op: MlirOperation) {
    unsafe { cleaveApplyNoInline(op) }
}

/// Turns each memory-to-memory copy of an aggregate of at least `min_bytes`
/// written as a load, `extractvalue`s and a store into a `memcpy`
/// (`cpp/shim.cpp`'s `cleaveCopyAggregatesInMemory`), where provably
/// equivalent. Returns the number rewritten. Run on the LLVM dialect.
///
/// # Safety
///
/// `op` must be a valid `builtin.module`, not used concurrently.
pub unsafe fn copy_aggregates_in_memory(op: MlirOperation, min_bytes: i64) -> i64 {
    unsafe { cleaveCopyAggregatesInMemory(op, min_bytes) }
}

/// Borrows `s`'s own bytes -- the C++ side only ever reads this synchronously
/// during the call it's passed to, so no ownership/lifetime story beyond the
/// call itself is needed (matches `melior::string_ref::StringRef::new`'s own
/// same-shaped contract, reimplemented here rather than pulling in `melior`
/// as a whole for one helper).
fn str_ref(s: &str) -> MlirStringRef {
    MlirStringRef {
        data: s.as_ptr() as *const _,
        length: s.len(),
    }
}

/// An `mlir::ExecutionEngine`, built with a real, explicit CPU/feature-set
/// `TargetMachine` instead of whatever `JITTargetMachineBuilder::detectHost`
/// finds. Mirrors `melior::ExecutionEngine`'s own public surface (`lookup`/
/// `invoke_packed`/`register_symbol`/`dump_to_object_file`) so a caller
/// already using that type has nothing new to learn -- just a different
/// constructor.
pub struct ExecutionEngine {
    raw: MlirExecutionEngine,
}

impl ExecutionEngine {
    /// `target_cpu`/`target_features` -- pass `""` for either to keep
    /// `detectHost`'s own untouched behaviour for that one axis. The two
    /// non-empty cases match exactly how `stamp_target_cpu` (`cleave/src/
    /// pipeline.rs`) already formats these two values as real `llvm.func`
    /// attributes today (a plain CPU name; a comma-separated `+feature`/
    /// `-feature` list) -- this is the same string, just finally reaching
    /// somewhere that has a real effect on the compiled code.
    pub fn new(
        module: MlirModule,
        optimization_level: usize,
        shared_library_paths: &[&str],
        enable_object_dump: bool,
        enable_pic: bool,
        target_cpu: &str,
        target_features: &str,
        loop_unroll: bool,
    ) -> Self {
        let paths: Vec<MlirStringRef> = shared_library_paths.iter().map(|s| str_ref(s)).collect();
        let raw = unsafe {
            cleaveExecutionEngineCreateWithTarget(
                module,
                optimization_level as i32,
                paths.len() as i32,
                paths.as_ptr(),
                enable_object_dump,
                enable_pic,
                str_ref(target_cpu),
                str_ref(target_features),
                loop_unroll,
            )
        };
        Self { raw }
    }

    pub fn lookup(&self, name: &str) -> *mut () {
        unsafe { mlirExecutionEngineLookup(self.raw, str_ref(name)) as *mut () }
    }

    /// # Safety
    ///
    /// Same contract as `melior::ExecutionEngine::invoke_packed`: `arguments`
    /// must be valid, aligned pointers to the real argument/result storage
    /// the named function expects.
    pub unsafe fn invoke_packed(
        &self,
        name: &str,
        arguments: &mut [*mut ()],
    ) -> Result<(), InvokeError> {
        let result = unsafe {
            mlirExecutionEngineInvokePacked(self.raw, str_ref(name), arguments.as_mut_ptr() as _)
        };
        if result.value != 0 {
            Ok(())
        } else {
            Err(InvokeError)
        }
    }

    /// # Safety
    ///
    /// `ptr` must stay valid for the lifetime of `self` -- the JIT'd code
    /// may call through it at any point until this engine is dropped.
    pub unsafe fn register_symbol(&self, name: &str, ptr: *mut ()) {
        unsafe { mlirExecutionEngineRegisterSymbol(self.raw, str_ref(name), ptr as _) }
    }

    pub fn dump_to_object_file(&self, path: &str) {
        unsafe { mlirExecutionEngineDumpToObjectFile(self.raw, str_ref(path)) }
    }
}

impl Drop for ExecutionEngine {
    fn drop(&mut self) {
        unsafe { mlirExecutionEngineDestroy(self.raw) }
    }
}

/// Mirrors `melior::Error::InvokeFunction`'s own role -- a plain marker,
/// carrying no extra detail beyond "the JIT'd call itself reported failure",
/// matching what `mlirExecutionEngineInvokePacked`'s own `MlirLogicalResult`
/// return value carries.
#[derive(Debug)]
pub struct InvokeError;

impl std::fmt::Display for InvokeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "failed to invoke packed function")
    }
}

impl std::error::Error for InvokeError {}
