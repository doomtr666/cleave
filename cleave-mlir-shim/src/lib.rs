//! cleave's C API over MLIR and LLVM (`cpp/shim.cpp`, `doc/plan-mlir-shim.md`):
//! the target code is generated for (`Target`), an object file
//! (`emit_object`), a JIT (`ExecutionEngine`), and the IR rewrites cleave's
//! pipeline runs.

use mlir_sys::{
    MlirContext, MlirExecutionEngine, MlirModule, MlirStringRef, mlirExecutionEngineDestroy,
    mlirExecutionEngineInvokePacked, mlirExecutionEngineLookup, mlirExecutionEngineRegisterSymbol,
};

/// `cpp/shim.cpp`'s `CleaveTarget`, opaque here.
#[repr(C)]
struct RawTarget {
    _private: [u8; 0],
}

/// Receives an error message from the shim: `user_data` is a `&mut String`.
type ErrorCallback =
    unsafe extern "C" fn(*const std::os::raw::c_char, usize, *mut std::ffi::c_void);

unsafe extern "C" fn keep_error(
    message: *const std::os::raw::c_char,
    len: usize,
    user_data: *mut std::ffi::c_void,
) {
    // SAFETY: the shim passes `len` bytes at `message`, and `user_data` is the
    // `&mut String` its caller handed over (`with_error`).
    unsafe {
        let bytes = std::slice::from_raw_parts(message as *const u8, len);
        *(user_data as *mut String) = String::from_utf8_lossy(bytes).into_owned();
    }
}

/// Calls `f` with the shim's error callback and its user data, returning
/// what `f` returns and the message the shim reported, if any.
fn with_error<T>(f: impl FnOnce(ErrorCallback, *mut std::ffi::c_void) -> T) -> (T, String) {
    let mut message = String::new();
    let result = f(
        keep_error,
        &mut message as *mut String as *mut std::ffi::c_void,
    );
    (result, message)
}

unsafe extern "C" {
    fn cleaveTargetCreate(
        cpu: MlirStringRef,
        features: MlirStringRef,
        opt_level: i32,
        pic: bool,
        loop_unroll: bool,
        on_error: ErrorCallback,
        user_data: *mut std::ffi::c_void,
    ) -> *mut RawTarget;
    fn cleaveTargetDestroy(target: *mut RawTarget);
    fn cleaveEmitObject(
        module: MlirModule,
        target: *mut RawTarget,
        path: MlirStringRef,
        on_error: ErrorCallback,
        user_data: *mut std::ffi::c_void,
    ) -> bool;
    fn cleaveJitCreate(
        module: MlirModule,
        target: *mut RawTarget,
        num_paths: i32,
        shared_lib_paths: *const MlirStringRef,
        on_error: ErrorCallback,
        user_data: *mut std::ffi::c_void,
    ) -> MlirExecutionEngine;
    fn cleaveRegisterPasses();
    fn cleaveRunPipeline(
        module: MlirModule,
        pipeline: MlirStringRef,
        statistics: bool,
        on_error: ErrorCallback,
        user_data: *mut std::ffi::c_void,
    ) -> bool;
    fn cleaveLoadTransformLibrary(context: MlirContext, text: MlirStringRef, name: MlirStringRef) -> bool;
}

/// Registers cleave's passes and MLIR's, for pipelines to name them
/// (`cpp/shim.cpp`: `cleave-elide-block-copies`, `cleave-lower-spawns{tasks=..}`,
/// ...). Idempotent; `run_pipeline` calls it.
pub fn register_passes() {
    unsafe { cleaveRegisterPasses() }
}

/// Runs the textual pipeline `pipeline` (`builtin.module(...)`) on `module`,
/// printing the pass statistics on stderr if `statistics` (what cleave's own
/// passes rewrote). An error for a pipeline that doesn't parse; a failing
/// pass reports through the context's diagnostic handlers and returns
/// `Err` with an empty message.
///
/// # Safety
///
/// `module` must be a valid module, not used concurrently.
pub unsafe fn run_pipeline(module: MlirModule, pipeline: &str, statistics: bool) -> Result<(), String> {
    let (ok, message) = with_error(|on_error, user_data| unsafe {
        cleaveRunPipeline(module, str_ref(pipeline), statistics, on_error, user_data)
    });
    if ok { Ok(()) } else { Err(message) }
}

/// Loads the transform module `text` into the transform dialect's library for
/// `context`, for `transform-interpreter` to find the sequences it names:
/// `transform-preload-library`, from memory. `name` names it in diagnostics.
/// `false` on error, reported through the context's diagnostic handlers.
///
/// # Safety
///
/// `context` must be a valid context, not used concurrently.
pub unsafe fn load_transform_library(context: MlirContext, text: &str, name: &str) -> bool {
    unsafe { cleaveLoadTransformLibrary(context, str_ref(text), str_ref(name)) }
}

fn str_ref(s: &str) -> MlirStringRef {
    MlirStringRef {
        data: s.as_ptr() as *const _,
        length: s.len(),
    }
}

/// The machine code is generated for and how it is optimized: the host's,
/// unless a CPU (`native`: the host's, with every feature it has) or a
/// comma-separated `+f`/`-f` feature list says otherwise. Built once, used
/// for objects (`emit_object`) and JITs (`ExecutionEngine`).
pub struct Target {
    raw: *mut RawTarget,
}

impl Target {
    /// `opt_level` 0 to 3; `loop_unroll` lets LLVM unroll loops. An unknown
    /// CPU or feature list is an error.
    pub fn new(
        cpu: Option<&str>,
        features: Option<&str>,
        opt_level: usize,
        pic: bool,
        loop_unroll: bool,
    ) -> Result<Self, String> {
        let (raw, message) = with_error(|on_error, user_data| unsafe {
            cleaveTargetCreate(
                str_ref(cpu.unwrap_or("")),
                str_ref(features.unwrap_or("")),
                opt_level as i32,
                pic,
                loop_unroll,
                on_error,
                user_data,
            )
        });
        if raw.is_null() {
            Err(message)
        } else {
            Ok(Self { raw })
        }
    }
}

impl Drop for Target {
    fn drop(&mut self) {
        unsafe { cleaveTargetDestroy(self.raw) }
    }
}

/// Writes `module`, in the LLVM dialect, as an object file at `path`,
/// compiled for `target`. Its external symbols are left for the linker.
///
/// # Safety
///
/// `module` must be a valid module, not used concurrently.
pub unsafe fn emit_object(module: MlirModule, target: &Target, path: &str) -> Result<(), String> {
    let (ok, message) = with_error(|on_error, user_data| unsafe {
        cleaveEmitObject(module, target.raw, str_ref(path), on_error, user_data)
    });
    if ok { Ok(()) } else { Err(message) }
}

/// A JIT (`mlir::ExecutionEngine`) for a module in the LLVM dialect.
pub struct ExecutionEngine {
    raw: MlirExecutionEngine,
}

impl ExecutionEngine {
    /// Compiles `module` for `target`, loading `shared_library_paths` for
    /// the symbols they define.
    ///
    /// # Safety
    ///
    /// `module` must be a valid module, not used concurrently.
    pub unsafe fn new(
        module: MlirModule,
        target: &Target,
        shared_library_paths: &[&str],
    ) -> Result<Self, String> {
        let paths: Vec<MlirStringRef> = shared_library_paths.iter().map(|s| str_ref(s)).collect();
        let (raw, message) = with_error(|on_error, user_data| unsafe {
            cleaveJitCreate(
                module,
                target.raw,
                paths.len() as i32,
                paths.as_ptr(),
                on_error,
                user_data,
            )
        });
        if raw.ptr.is_null() {
            Err(message)
        } else {
            Ok(Self { raw })
        }
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
