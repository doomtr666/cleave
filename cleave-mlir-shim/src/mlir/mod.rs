//! cleave's MLIR API: building and inspecting IR over MLIR's C API (`sys`),
//! with the shapes and names cleave's lowering was written against
//! (`ir::Block`, `ir::operation::OperationBuilder`, `dialect::llvm::load`,
//! ...). Owned objects (`Context`, `Module`, `Operation`, `Block`, `Region`
//! not yet inserted anywhere) are destroyed when dropped; the `...Ref` types
//! borrow ones their parent owns.

pub mod sys;

use std::ffi::c_void;
use std::fmt;
use std::marker::PhantomData;

/// A borrowed UTF-8 string for the C API.
#[derive(Clone, Copy)]
pub struct StringRef<'a> {
    raw: sys::MlirStringRef,
    _string: PhantomData<&'a str>,
}

impl<'a> StringRef<'a> {
    pub fn new(string: &'a str) -> Self {
        Self { raw: sys::MlirStringRef { data: string.as_ptr() as _, length: string.len() }, _string: PhantomData }
    }

    /// # Safety
    ///
    /// `raw` must point to `length` bytes living as long as `'a`.
    pub unsafe fn from_raw(raw: sys::MlirStringRef) -> Self {
        Self { raw, _string: PhantomData }
    }

    pub fn to_raw(self) -> sys::MlirStringRef {
        self.raw
    }

    pub fn as_str(&self) -> Result<&'a str, std::str::Utf8Error> {
        if self.raw.data.is_null() {
            return Ok("");
        }
        // SAFETY: the C API hands out `length` bytes at `data`, alive for `'a`.
        std::str::from_utf8(unsafe { std::slice::from_raw_parts(self.raw.data as *const u8, self.raw.length) })
    }
}

/// An error from the C API: what it reported, or what it returned null for.
#[derive(Debug, Clone)]
pub struct Error(pub String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

/// Collects what a C API printer writes into a `String`.
unsafe extern "C" fn print_to_string(text: sys::MlirStringRef, user_data: *mut c_void) {
    // SAFETY: `user_data` is the `String` `printed` passes; `text` is valid
    // for the call.
    unsafe {
        let out = &mut *(user_data as *mut String);
        let bytes = std::slice::from_raw_parts(text.data as *const u8, text.length);
        out.push_str(&String::from_utf8_lossy(bytes));
    }
}

/// What `print` (a C API printer) writes.
fn printed(print: impl FnOnce(sys::MlirStringCallback, *mut c_void)) -> String {
    let mut out = String::new();
    print(print_to_string, &mut out as *mut String as *mut c_void);
    out
}

/// An MLIR context: owns every type, attribute and location made in it.
pub struct Context {
    raw: sys::MlirContext,
}

impl Context {
    pub fn new() -> Self {
        Self { raw: unsafe { sys::mlirContextCreate() } }
    }

    pub fn to_raw(&self) -> sys::MlirContext {
        self.raw
    }

    pub fn append_dialect_registry(&self, registry: &dialect::DialectRegistry) {
        unsafe { sys::mlirContextAppendDialectRegistry(self.raw, registry.to_raw()) }
    }

    pub fn load_all_available_dialects(&self) {
        unsafe { sys::mlirContextLoadAllAvailableDialects(self.raw) }
    }

    pub fn set_allow_unregistered_dialects(&self, allow: bool) {
        unsafe { sys::mlirContextSetAllowUnregisteredDialects(self.raw, allow) }
    }

    /// Calls `handler` with each diagnostic; one it returns `true` for is
    /// handled, the others go on to the handlers attached before it.
    pub fn attach_diagnostic_handler<F: FnMut(Diagnostic) -> bool + 'static>(&self, handler: F) -> DiagnosticHandlerId {
        unsafe extern "C" fn call<F: FnMut(Diagnostic) -> bool>(
            diagnostic: sys::MlirDiagnostic,
            user_data: *mut c_void,
        ) -> sys::MlirLogicalResult {
            // SAFETY: `user_data` is the boxed `F` attached below.
            let handler = unsafe { &mut *(user_data as *mut F) };
            sys::MlirLogicalResult { value: handler(Diagnostic { raw: diagnostic, _context: PhantomData }) as i8 }
        }
        unsafe extern "C" fn delete<F>(user_data: *mut c_void) {
            // SAFETY: `user_data` is the box leaked below, freed once.
            drop(unsafe { Box::from_raw(user_data as *mut F) });
        }
        let user_data = Box::into_raw(Box::new(handler)) as *mut c_void;
        DiagnosticHandlerId(unsafe {
            sys::mlirContextAttachDiagnosticHandler(self.raw, call::<F>, user_data, Some(delete::<F>))
        })
    }

    pub fn detach_diagnostic_handler(&self, id: DiagnosticHandlerId) {
        unsafe { sys::mlirContextDetachDiagnosticHandler(self.raw, id.0) }
    }
}

impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe { sys::mlirContextDestroy(self.raw) }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DiagnosticHandlerId(sys::MlirDiagnosticHandlerID);

/// A diagnostic, valid for the duration of its handler's call.
pub struct Diagnostic<'a> {
    raw: sys::MlirDiagnostic,
    _context: PhantomData<&'a ()>,
}

impl fmt::Display for Diagnostic<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(&printed(|cb, ud| unsafe { sys::mlirDiagnosticPrint(self.raw, cb, ud) }))
    }
}

pub mod ir;
pub mod dialect;

pub mod utility {
    use super::{Context, dialect::DialectRegistry, sys};

    pub fn register_all_dialects(registry: &DialectRegistry) {
        unsafe { sys::mlirRegisterAllDialects(registry.to_raw()) }
    }

    pub fn register_all_llvm_translations(context: &Context) {
        unsafe { sys::mlirRegisterAllLLVMTranslations(context.to_raw()) }
    }
}
