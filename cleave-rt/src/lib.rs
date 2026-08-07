//! The Rust-implemented half of cleave's `extern fn` stdlib. Each function
//! here is compiled directly into the `cleave` binary itself (an ordinary
//! path dependency, not a separately loaded shared library) and registered
//! with the JIT `ExecutionEngine` by real function pointer (see `main.rs`'s
//! `--run` path) — no dynamic symbol lookup by name involved, sidestepping
//! the Windows/MSVC CRT-symbol-visibility questions a raw libc binding would
//! have run into.
//!
//! `extern "C"` on each function is the calling-convention marker MLIR's own
//! generated `func.call`/`llvm.call` needs to match; it's required
//! regardless of how the pointer reaches the JIT.

pub extern "C" fn print_i8(x: i8) -> i8 {
    println!("{x}");
    x
}

pub extern "C" fn print_i16(x: i16) -> i16 {
    println!("{x}");
    x
}

pub extern "C" fn print_i32(x: i32) -> i32 {
    println!("{x}");
    x
}

pub extern "C" fn print_i64(x: i64) -> i64 {
    println!("{x}");
    x
}

pub extern "C" fn print_f32(x: f32) -> f32 {
    println!("{x}");
    x
}

pub extern "C" fn print_f64(x: f64) -> f64 {
    println!("{x}");
    x
}

/// Backs every struct construction (`mlir_lower.rs::alloc_struct`) — a
/// struct is a stable, heap-backed reference (mutated in place, passed
/// around and returned by pointer, never copied field-by-field — see
/// `mlir_lower.rs::struct_llvm_type`'s own doc comment), so its own storage
/// must outlive the function that constructs it: `llvm.alloca` (stack)
/// doesn't, found by direct testing (a struct returned from one function and
/// read by its caller came back reading garbage/reused stack memory once
/// heap allocation wasn't yet in place). Deliberately leaks — cleave has no
/// `drop`/ownership story yet, matching this project's current "no memory
/// management design yet" scope; not a bug to fix here, a real gap to
/// revisit once one exists.
pub extern "C" fn cleave_alloc(size: i64) -> *mut u8 {
    let layout = std::alloc::Layout::from_size_align(size as usize, 16).expect("cleave_alloc: invalid layout");
    unsafe { std::alloc::alloc(layout) }
}
