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
//!
//! `#[unsafe(no_mangle)]` on every one: irrelevant to the JIT path (which registers
//! by real function pointer, never by name lookup), but required once a real
//! `.o`/staticlib is linked by an external linker (`--emit-object`, Axis
//! B/A) — without it, Rust's own name-mangling means no `extern fn`/
//! `export fn` call site anywhere could actually resolve against the real
//! symbol by its plain name.

#[unsafe(no_mangle)]
pub extern "C" fn print_i8(x: i8) -> i8 {
    println!("{x}");
    x
}

#[unsafe(no_mangle)]
pub extern "C" fn print_i16(x: i16) -> i16 {
    println!("{x}");
    x
}

#[unsafe(no_mangle)]
pub extern "C" fn print_i32(x: i32) -> i32 {
    println!("{x}");
    x
}

#[unsafe(no_mangle)]
pub extern "C" fn print_i64(x: i64) -> i64 {
    println!("{x}");
    x
}

#[unsafe(no_mangle)]
pub extern "C" fn print_f32(x: f32) -> f32 {
    println!("{x}");
    x
}

#[unsafe(no_mangle)]
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
#[unsafe(no_mangle)]
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
#[unsafe(no_mangle)]
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
#[unsafe(no_mangle)]
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
#[unsafe(no_mangle)]
pub unsafe extern "C" fn print_bytes(ptr: *const u8, len: i64) -> i64 {
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len as usize) };
    use std::io::Write;
    std::io::stdout().write_all(bytes).expect("print_bytes: stdout write failed");
    len
}

/// Writes `len` bytes starting at `buf` to stdout as raw text -- `Print<T>`'s
/// own scalar/string impls (`stdlib/io/io.cleave`) each hardcode their own
/// backing extern (`print_i32`, `print_bytes`, ...); the new `Display<T>`-
/// backed impls (arrays/tensors/tuples of a `Display`-able element type,
/// `stdlib/display/display.cleave`) all build one `DynArray<i8>` buffer the
/// identical way regardless of the underlying type, so they share this one
/// flush primitive instead of each declaring their own. Structurally
/// identical to `print_bytes` above -- the only real difference is the
/// argument shape: a `DynArray<i8>`'s own `buf` field is already a bare,
/// opaque pointer by construction (`RawBuf`'s own doc comment,
/// `stdlib/dynarray/dynarray.cleave`), not an array-typed value needing
/// `mlir_lower.rs`'s own array-aware `(ptr,len)` extraction the way
/// `print_bytes`'s own `[i8;N]` argument does -- passed straight through as
/// an ordinary opaque-pointer argument instead.
///
/// # Safety
/// `buf` must point to at least `len` readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn print_dynarray_bytes(buf: *const u8, len: i32) -> i32 {
    let bytes = unsafe { std::slice::from_raw_parts(buf, len as usize) };
    use std::io::Write;
    std::io::stdout()
        .write_all(bytes)
        .expect("print_dynarray_bytes: stdout write failed");
    len
}

/// Writes `x`'s own decimal `Display` form into `out` (at least 24 bytes,
/// enough for any `f32`/`f64` this project's own `format!("{x}")` -- the
/// *same* formatting `print_f32`/`print_f64` above already use, so a float
/// reads identically whether it reached stdout via the old direct `Print<T>`
/// path or the new `Display<T>`-composed one -- ever produces), returns the
/// real byte count written. Backs `Display<f32>`/`Display<f64>`
/// (`stdlib/display/display.cleave`) -- the one part of `Display<T>` that
/// genuinely needs a real extern rather than being expressible in ordinary
/// cleave source (integer digit-extraction is plain arithmetic, easy to
/// write by hand in cleave; a correct, shortest-round-trip float-to-decimal
/// algorithm is not something to hand-reimplement). Confirmed directly, not
/// assumed, that writing through an array-typed extern *argument* (rather
/// than only ever reading one, every prior array-argument extern in this
/// codebase's own precedent) is actually visible to cleave code once the
/// call returns -- a throwaway probe extern, since removed, wrote known
/// bytes into a `[i8;4]` argument and cleave correctly read them back.
///
/// Rust's own unadorned `{}` float `Display` never uses scientific
/// notation, so a genuinely extreme value (a denormal near `f64::MIN_
/// POSITIVE`, say) can format to *far* more than 24 bytes -- ordinary
/// training/inference values (this project's own actual use so far) never
/// come close, so the buffer stays small for that overwhelmingly common
/// case; an extreme value is truncated to `CAP` bytes rather than
/// overflowing the caller's buffer or panicking the whole JIT session, a
/// real but accepted, honestly-not-round-trippable v1 limitation.
///
/// # Safety
/// `out` must point to at least `CAP` (24) writable bytes.
macro_rules! format_float {
    ($name:ident, $ty:ty) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $name(x: $ty, out: *mut u8) -> i32 {
            const CAP: usize = 24;
            let s = format!("{x}");
            let bytes = &s.as_bytes()[..s.len().min(CAP)];
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), out, bytes.len()) };
            bytes.len() as i32
        }
    };
}
format_float!(format_f32, f32);
format_float!(format_f64, f64);

/// Backs `stdlib/dynarray/dynarray.cleave`'s `DynArray<T>` -- a real,
/// growable collection (`doc/backlog.md`'s own former "No dynamic-size
/// collection" item), built entirely as an ordinary stdlib struct + algebra
/// impls, the same "no new `Ty::Vector`-style compiler variant" discipline
/// `stdlib/linalg/tensor.cleave`'s own top comment already documents.
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
#[unsafe(no_mangle)]
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
        #[unsafe(no_mangle)]
        pub extern "C" fn $alloc(cap: i32) -> *mut $elem {
            unsafe { cleave_realloc(std::ptr::null_mut(), 0, cap as i64 * std::mem::size_of::<$elem>() as i64) as *mut $elem }
        }
        #[unsafe(no_mangle)]
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
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $get(buf: *const $elem, i: i32) -> $elem {
            unsafe { *buf.add(i as usize) }
        }
        /// # Safety
        /// Same as `$get` above.
        #[unsafe(no_mangle)]
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

/// MLIR's own `memref.copy` runtime helper (`mlir::ExecutionEngine::
/// CRunnerUtils.h`'s own `memrefCopy`), reimplemented here rather than
/// loaded from the real `mlir_c_runner_utils.dll` (`I:/Dev/llvm-mlir-22`'s
/// own real MLIR 22 build, confirmed to genuinely export it via `dumpbin /
/// exports` -- not a stub or a missing build) -- `one-shot-bufferize`'s own
/// generated `memref.copy` calls need it the moment a tensor value is big
/// enough to need a real defensive copy before a write (`Dense`/`Network`,
/// `examples/xor_tensor.cleave`, is the first cleave program ever to trigger
/// one — every prior example's own lowered IR has zero `memref.copy` calls
/// at all, confirmed directly via `--dump-mlir-lowered`), and this project's
/// own JIT (`melior::ExecutionEngine::new`) never had any shared library
/// loaded alongside the lowered module to satisfy it. Loading the real DLL
/// was tried first and abandoned: passing more than one path in `melior`'s
/// own `shared_library_paths` array (needed since `mlir_c_runner_utils.dll`
/// itself depends on the sibling `mlir_float16_utils.dll`, confirmed via
/// `dumpbin /dependents`, and `I:/Dev/llvm-mlir-22/bin` was never on this
/// process's own DLL search path) corrupted the *first* path into the
/// second with no separator between them (`Failed to create MemoryBuffer
/// for: ...dllI:/Dev/...`) -- a real bug somewhere in `melior`'s/MLIR's own
/// C API glue around a non-null-terminated `MlirStringRef`, not this
/// project's own code, and not worth chasing further versus just owning a
/// small, correct reimplementation here instead — matches this crate's own
/// existing posture (`dynarray_*` above already reimplements, rather than
/// links against, the array-growth runtime a real language would often
/// pull from an external allocator library).
///
/// ABI, read directly from the real header (no `.cpp` shipped alongside it,
/// only headers -- this project's own MLIR 22 install is headers + prebuilt
/// libs, not full source) -- `UnrankedMemRefType<char>{ rank: i64, descriptor
/// : *mut c_void }`, `descriptor` pointing to a `{ basePtr, data, offset,
/// sizes[rank], strides[rank] }` ranked-memref descriptor (the ordinary
/// `memref`-to-`llvm` ABI this project's own `mlir_lower.rs` already
/// produces everywhere else) -- `sizes`/`strides` read directly as raw byte
/// offsets into `descriptor` rather than through a typed Rust struct: their
/// own length depends on `rank`, a *runtime* value, not expressible as an
/// ordinary fixed-layout `#[repr(C)]` struct field.
///
/// A plain element-by-element strided copy, not the real implementation's
/// own presumably-more-optimized one (a contiguous-run fast path, likely) --
/// semantically identical either way, and every shape this project's own
/// `Tensor<T,Dims...>` ever produces is tiny (single-digit element counts),
/// so the performance difference is immaterial here.
#[repr(C)]
pub struct UnrankedMemRef {
    rank: i64,
    descriptor: *mut u8,
}

/// # Safety
/// `src`/`dst` must each point to a live `UnrankedMemRef` whose own
/// `descriptor` points to a real ranked-memref descriptor of that same
/// struct's own `rank`, exactly the shape MLIR's own `memref.copy` lowering
/// always produces -- never called directly from cleave source, only ever
/// invoked by the JIT itself, on `mlir_lower.rs`'s own generated code.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn memrefCopy(elem_size: i64, src: *const UnrankedMemRef, dst: *const UnrankedMemRef) {
    unsafe {
        let elem_size = elem_size as usize;
        let rank = (*src).rank as usize;
        let src_desc = (*src).descriptor;
        let dst_desc = (*dst).descriptor;

        let read_i64 = |base: *mut u8, byte_off: usize| -> i64 {
            std::ptr::read_unaligned(base.add(byte_off) as *const i64)
        };
        let read_ptr =
            |base: *mut u8, byte_off: usize| -> *mut u8 { std::ptr::read_unaligned(base.add(byte_off) as *const *mut u8) };

        // Layout: `basePtr: *mut u8` (8 bytes, unused here), `data: *mut u8`
        // (8), `offset: i64` (8), then `sizes`/`strides`, each `rank` `i64`s.
        let src_data = read_ptr(src_desc, 8);
        let dst_data = read_ptr(dst_desc, 8);
        let src_offset = read_i64(src_desc, 16);
        let dst_offset = read_i64(dst_desc, 16);
        let sizes_off = 24;
        let strides_off = 24 + rank * 8;

        if rank == 0 {
            std::ptr::copy_nonoverlapping(
                src_data.add(src_offset as usize * elem_size),
                dst_data.add(dst_offset as usize * elem_size),
                elem_size,
            );
            return;
        }

        let sizes: Vec<i64> = (0..rank).map(|i| read_i64(src_desc, sizes_off + i * 8)).collect();
        let src_strides: Vec<i64> = (0..rank).map(|i| read_i64(src_desc, strides_off + i * 8)).collect();
        let dst_strides: Vec<i64> = (0..rank).map(|i| read_i64(dst_desc, strides_off + i * 8)).collect();

        let total: i64 = sizes.iter().product();
        let mut indices = vec![0i64; rank];
        for _ in 0..total {
            let mut src_off = src_offset;
            let mut dst_off = dst_offset;
            for i in 0..rank {
                src_off += indices[i] * src_strides[i];
                dst_off += indices[i] * dst_strides[i];
            }
            std::ptr::copy_nonoverlapping(
                src_data.add(src_off as usize * elem_size),
                dst_data.add(dst_off as usize * elem_size),
                elem_size,
            );
            for i in (0..rank).rev() {
                indices[i] += 1;
                if indices[i] < sizes[i] {
                    break;
                }
                indices[i] = 0;
            }
        }
    }
}
dynarray_width!(*mut u8, dynarray_alloc_ptr, dynarray_grow_ptr, dynarray_get_ptr, dynarray_set_ptr);

/// A minimal PRNG for `stdlib/rand/rand.cleave` — PCG32 (O'Neill, public
/// domain), the "one-sequence" variant: a single 64-bit state, advanced by a
/// fixed linear congruential step, output-permuted through an xorshift +
/// variable rotation to hide the LCG's own well-known low-bit weakness. No
/// new dependency (`cleave-rt/Cargo.toml` has none at all today) -- the
/// same reasoning that led to hand-reimplementing `memrefCopy` above rather
/// than loading a real DLL: this is a small, public, easily-verified-by-hand
/// algorithm, not worth a crate for. `Ordering::Relaxed` throughout -- this
/// runtime is already implicitly single-threaded everywhere else (every
/// other piece of mutable state here, `cleave_alloc`'s own allocator
/// included, assumes the same).
static PCG_STATE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0x853c_49e6_748f_ea9b);
const PCG_MULT: u64 = 6364136223846793005;
const PCG_INC: u64 = 1442695040888963407;

fn pcg32_next_u32() -> u32 {
    let old = PCG_STATE.load(std::sync::atomic::Ordering::Relaxed);
    let new = old.wrapping_mul(PCG_MULT).wrapping_add(PCG_INC);
    PCG_STATE.store(new, std::sync::atomic::Ordering::Relaxed);
    let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
    let rot = (old >> 59) as u32;
    xorshifted.rotate_right(rot)
}

/// Reseeds the global generator -- `s` becomes the PRNG's own next `state`
/// directly (no separate stream/sequence parameter, matching `PCG_INC`
/// being a fixed constant above). Two calls with the same seed reproduce
/// the exact same following sequence, by construction.
#[unsafe(no_mangle)]
pub extern "C" fn rand_seed(s: i64) {
    PCG_STATE.store(s as u64, std::sync::atomic::Ordering::Relaxed);
}

/// Canonical uniform `[0,1)` -- the standard "top N mantissa bits of a raw
/// word, divided by 2^N" construction: every representable output is
/// exactly reachable and uniformly likely, no rounding bias at the
/// boundaries. `f32` has a 24-bit mantissa (23 explicit + the implicit
/// leading 1), so the top 24 bits of one `pcg32_next_u32()` draw are
/// exactly enough.
#[unsafe(no_mangle)]
pub extern "C" fn rand_uniform_f32() -> f32 {
    (pcg32_next_u32() >> 8) as f32 * (1.0 / (1u32 << 24) as f32)
}

/// Same construction as `rand_uniform_f32`, scaled up to `f64`'s own 53-bit
/// mantissa -- one `pcg32_next_u32()` draw alone is short of that, so two
/// draws are combined into one 64-bit word first.
#[unsafe(no_mangle)]
pub extern "C" fn rand_uniform_f64() -> f64 {
    let hi = pcg32_next_u32() as u64;
    let lo = pcg32_next_u32() as u64;
    let combined = (hi << 32) | lo;
    (combined >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
}

/// Standard normal `N(0,1)` via the Box-Muller transform, consuming two
/// independent uniform draws per call. `u1` is floored at a tiny epsilon --
/// `rand_uniform_f32`'s own `[0,1)` range includes exactly `0.0`, and
/// `ln(0.0)` is `-inf` -- astronomically unlikely (1 in 2^24) but a real,
/// cheap-to-avoid edge case, not worth leaving in.
#[unsafe(no_mangle)]
pub extern "C" fn rand_normal_f32() -> f32 {
    let u1 = rand_uniform_f32().max(1e-7);
    let u2 = rand_uniform_f32();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

/// `f64` counterpart of `rand_normal_f32`, same construction.
#[unsafe(no_mangle)]
pub extern "C" fn rand_normal_f64() -> f64 {
    let u1 = rand_uniform_f64().max(1e-15);
    let u2 = rand_uniform_f64();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

#[cfg(test)]
mod rand_tests {
    use super::*;

    /// Every RNG behavior check lives in *one* `#[test]` deliberately --
    /// `PCG_STATE` is one process-wide global, and `cargo test` runs
    /// `#[test]` functions on separate threads *in parallel* by default;
    /// found directly, by testing: splitting these into several separate
    /// tests raced on that shared state (one test's own `rand_seed` call
    /// landing between another's own seed and its first draw), an
    /// intermittent failure with no bug in the RNG itself. One test means
    /// one thread, no race -- the same fix as making any global-state test
    /// sequential, not specific to this RNG.
    #[test]
    fn pcg32_behaves_correctly() {
        // PCG32's own reference sequence for state `42`, `inc` fixed to
        // `PCG_INC` above -- computed independently against the public PCG
        // minimal-C reference implementation (one LCG step from `old = 42`,
        // then the xorshift+rotate output permutation), not just re-derived
        // from this same Rust code -- a real cross-check, not a tautology.
        // `first == 0` is not a bug: `42`'s own top bits are all zero, and
        // the output permutation reads `old` *before* the LCG step mixes it,
        // so a small enough seed's very first output can legitimately be
        // `0` -- confirmed against the reference computation, not assumed.
        PCG_STATE.store(42, std::sync::atomic::Ordering::Relaxed);
        assert_eq!(pcg32_next_u32(), 0);
        assert_eq!(pcg32_next_u32(), 1971522493);
        assert_eq!(pcg32_next_u32(), 242089394);

        // Reproducibility: reseeding to the exact same state replays the
        // exact same sequence.
        PCG_STATE.store(42, std::sync::atomic::Ordering::Relaxed);
        assert_eq!(pcg32_next_u32(), 0);
        assert_eq!(pcg32_next_u32(), 1971522493);

        // `rand_seed` (the extern fn cleave itself calls) end to end, not
        // just the raw PCG32 step.
        rand_seed(1234);
        let a0 = rand_uniform_f32();
        let a1 = rand_uniform_f32();
        rand_seed(1234);
        let b0 = rand_uniform_f32();
        let b1 = rand_uniform_f32();
        assert_eq!(a0, b0);
        assert_eq!(a1, b1);
        assert_ne!(a0, a1);

        rand_seed(7);
        for _ in 0..1000 {
            let x = rand_uniform_f32();
            assert!((0.0..1.0).contains(&x), "{x} out of [0,1)");
        }

        rand_seed(99);
        let draws: Vec<f32> = (0..1000).map(|_| rand_normal_f32()).collect();
        let mean: f32 = draws.iter().sum::<f32>() / draws.len() as f32;
        // A real, if loose, sanity bound -- N(0,1)'s own sample mean over
        // 1000 draws should land well within +/-0.2 almost always; this is
        // not a statistical rigor test, just a guard against a broken
        // implementation returning something wildly non-normal (e.g.
        // always ~0, or unbounded).
        assert!(mean.abs() < 0.2, "sample mean {mean} too far from 0");
        assert!(draws.iter().any(|&x| x < -0.5));
        assert!(draws.iter().any(|&x| x > 0.5));
    }
}
