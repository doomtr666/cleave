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

/// De-risks the extern/array ABI boundary in isolation, before string
/// support depends on it (`cleave/tests/mlir_lower.rs::an_array_argument_
/// crosses_an_extern_call_boundary_correctly`) — a real `extern fn` taking
/// a `[i8; N]` array argument, matching `mlir_lower.rs`'s own array-aware
/// extern-call lowering (a raw pointer + a compile-time-known length,
/// passed as two ordinary scalar arguments, never the array's own `memref`
/// directly).
///
/// # Safety
/// `ptr` must point to at least `len` readable bytes — guaranteed by
/// construction: only `mlir_lower.rs`'s own array-to-extern-call lowering
/// ever calls this, always with a real array's own storage and its own
/// exact declared length.
pub unsafe extern "C" fn sum_bytes(ptr: *const i8, len: i64) -> i32 {
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len as usize) };
    bytes.iter().map(|&b| b as i32).sum()
}

/// De-risks a genuinely `void`-returning `extern fn` (`cleave/tests/
/// mlir_lower.rs::a_unit_returning_extern_fn_can_be_called_correctly`) — a
/// real C ABI shape `cleave-rt` had never exercised before: every other
/// `extern fn` here has a real, non-unit return value (`Print<T>`'s own
/// "prints and returns unchanged" contract, or `Print<[i8;N]>`'s own
/// discarded-`i64` reconciliation), so `mlir_lower.rs`'s `PrimOp::Extern`
/// lowering had never needed to declare/call an extern symbol with *zero*
/// results at all.
pub extern "C" fn touch_i32(_x: i32) {}

/// Writes `len` bytes starting at `ptr` to stdout as raw text — backs
/// `Print<[i8; N]>` (`stdlib/io/io.cleave`), mirrors `print_i32`'s own
/// "print and return unchanged" contract, but a `[i8; N]` argument reaches
/// this as a raw `(ptr, len)` pair, not a single scalar (see
/// `mlir_lower.rs`'s own array-aware extern-call lowering doc comment for
/// why: an MLIR `memref`'s default descriptor-struct calling convention has
/// no stable match to an ordinary C ABI, so `mlir_lower.rs` extracts a bare
/// pointer + a compile-time-known length explicitly before this call
/// rather than passing the memref itself).
///
/// # Safety
/// `ptr` must point to at least `len` readable bytes — guaranteed by
/// construction, same reasoning as `sum_bytes` above.
pub unsafe extern "C" fn print_bytes(ptr: *const u8, len: i64) -> i64 {
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len as usize) };
    use std::io::Write;
    std::io::stdout().write_all(bytes).expect("print_bytes: stdout write failed");
    len
}

/// Backs `stdlib/dynarray/dynarray.cleave`'s `DynArray<T>` -- a real,
/// growable collection (`doc/backlog.md`'s own former "No dynamic-size
/// collection" item), built entirely as an ordinary stdlib struct + algebra
/// impls, the same "no new `Ty::Vector`-style compiler variant" discipline
/// `stdlib/vector/vector.cleave`'s own top comment already documents.
///
/// The one shared, byte-count-based growth primitive -- every per-width
/// `dynarray_grow_*`/`dynarray_alloc_*` below (see `dynarray_width!` further
/// down) just converts its own element count to bytes and delegates here,
/// exactly the way `cleave_alloc` above is the one shared allocation
/// primitive every struct construction delegates to. `old_size == 0` means
/// "no real old block yet" (a fresh `DynArray`'s very first grow) --
/// `std::alloc::realloc` requires a pointer actually allocated with the
/// exact layout it's told, which doesn't exist yet in that case, so this
/// allocates fresh instead. Unlike `cleave_alloc` (deliberately leaked, no
/// free -- see its own doc comment above), a *real* grow (`old_size > 0`)
/// does not leak: `std::alloc::realloc` either extends the existing block in
/// place or moves the data and frees the old block itself. What's still
/// true, unchanged from every other struct in this codebase: a `DynArray`'s
/// own *final* buffer is never freed once the `DynArray` value itself is
/// discarded -- cleave has no `drop`/ownership story anywhere yet, not a new
/// gap this introduces.
pub unsafe extern "C" fn cleave_realloc(ptr: *mut u8, old_size: i64, new_size: i64) -> *mut u8 {
    let new_layout = std::alloc::Layout::from_size_align(new_size as usize, 16).expect("cleave_realloc: invalid layout");
    if old_size == 0 {
        unsafe { std::alloc::alloc(new_layout) }
    } else {
        let old_layout = std::alloc::Layout::from_size_align(old_size as usize, 16).expect("cleave_realloc: invalid layout");
        unsafe { std::alloc::realloc(ptr, old_layout, new_layout.size()) }
    }
}

/// Generates the four per-width raw-buffer primitives `RawBuffer<T>`'s own
/// per-width `impl` (`stdlib/dynarray/dynarray.cleave`) binds via
/// `extern(...)`: `alloc`/`grow` (element-count-based, converted to bytes
/// here, hardcoded per width -- exactly how `print_i32`/`print_f64`/...
/// above already hardcode their own width, no generic `sizeof` mechanism
/// needed anywhere) and `get`/`set` (plain pointer-offset read/write).
/// Invoked once per width below, including `*mut u8` -- the "any struct
/// element" case, since every cleave struct value is already an opaque
/// pointer of exactly this shape (`mlir_lower.rs::ty_to_mlir`'s own struct
/// fallback), so this one width's functions are reusable as-is by *any*
/// future struct-element `DynArray<Struct>`, no new Rust code needed per
/// struct type -- only a new `impl RawBuffer<Struct>` on the cleave side.
macro_rules! dynarray_width {
    ($elem:ty, $alloc:ident, $grow:ident, $get:ident, $set:ident) => {
        pub extern "C" fn $alloc(cap: i32) -> *mut $elem {
            unsafe { cleave_realloc(std::ptr::null_mut(), 0, cap as i64 * std::mem::size_of::<$elem>() as i64) as *mut $elem }
        }
        pub extern "C" fn $grow(old: *mut $elem, old_cap: i32, new_cap: i32) -> *mut $elem {
            unsafe {
                cleave_realloc(
                    old as *mut u8,
                    old_cap as i64 * std::mem::size_of::<$elem>() as i64,
                    new_cap as i64 * std::mem::size_of::<$elem>() as i64,
                ) as *mut $elem
            }
        }
        /// # Safety
        /// `buf` must point to a live buffer of at least `i + 1` `$elem`s --
        /// guaranteed by construction: only `DynArray<T>`'s own generated
        /// calls (`stdlib/dynarray/dynarray.cleave`) ever call this, always
        /// with its own real, currently-allocated buffer and an in-bounds
        /// index (no bounds checking, matching this codebase's existing
        /// "no runtime memory-safety enforcement" posture elsewhere).
        pub unsafe extern "C" fn $get(buf: *const $elem, i: i32) -> $elem {
            unsafe { *buf.add(i as usize) }
        }
        /// # Safety
        /// Same as `$get` above.
        pub unsafe extern "C" fn $set(buf: *mut $elem, i: i32, v: $elem) {
            unsafe {
                *buf.add(i as usize) = v;
            }
        }
    };
}

dynarray_width!(i8, dynarray_alloc_i8, dynarray_grow_i8, dynarray_get_i8, dynarray_set_i8);
dynarray_width!(i16, dynarray_alloc_i16, dynarray_grow_i16, dynarray_get_i16, dynarray_set_i16);
dynarray_width!(i32, dynarray_alloc_i32, dynarray_grow_i32, dynarray_get_i32, dynarray_set_i32);
dynarray_width!(i64, dynarray_alloc_i64, dynarray_grow_i64, dynarray_get_i64, dynarray_set_i64);
dynarray_width!(f32, dynarray_alloc_f32, dynarray_grow_f32, dynarray_get_f32, dynarray_set_f32);
dynarray_width!(f64, dynarray_alloc_f64, dynarray_grow_f64, dynarray_get_f64, dynarray_set_f64);
dynarray_width!(*mut u8, dynarray_alloc_ptr, dynarray_grow_ptr, dynarray_get_ptr, dynarray_set_ptr);
