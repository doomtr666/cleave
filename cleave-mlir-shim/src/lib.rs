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
    MlirExecutionEngine, MlirModule, MlirStringRef, mlirExecutionEngineDestroy,
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
    ) -> MlirExecutionEngine;
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
