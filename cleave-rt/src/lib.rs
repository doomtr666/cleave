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

// No trailing newline -- `print`/`Print<T>` (`stdlib/io/io.cleave`) writes
// exactly the bytes its argument's own decimal form is, nothing more, the
// same "operate, return unchanged" contract `print_bytes`/
// `print_dynarray_bytes` (below) already honor for a string/`Display`-built
// buffer. A caller wanting a trailing newline uses `println` (`stdlib/io/
// io.cleave`, a plain `T: Print`-bound wrapper -- `print(x); print(['\n']);`
// -- no separate runtime symbol needed for it at all). Found for real, not
// hypothetical: these used to hardcode `println!`, silently appending `\n`
// for *every* scalar while every string/array/tensor/tuple `Print<T>` impl
// (routed through `print_bytes`/`print_dynarray_bytes`, plain `write_all`,
// never `println!`) added none -- a genuine inconsistency, reported
// directly (`print("step "); print(step);` produced an invisible newline
// between them that wasn't written anywhere in the calling code).
#[unsafe(no_mangle)]
pub extern "C" fn print_i8(x: i8) -> i8 {
    print!("{x}");
    x
}

#[unsafe(no_mangle)]
pub extern "C" fn print_i16(x: i16) -> i16 {
    print!("{x}");
    x
}

#[unsafe(no_mangle)]
pub extern "C" fn print_i32(x: i32) -> i32 {
    print!("{x}");
    x
}

#[unsafe(no_mangle)]
pub extern "C" fn print_i64(x: i64) -> i64 {
    print!("{x}");
    x
}

#[unsafe(no_mangle)]
pub extern "C" fn print_f32(x: f32) -> f32 {
    print!("{x}");
    x
}

#[unsafe(no_mangle)]
pub extern "C" fn print_f64(x: f64) -> f64 {
    print!("{x}");
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
/// `drop`/ownership story yet. **The real fix, in progress**: `doc/hld.md`'s
/// own "Memory management" section — `cleave_alloc_rc`/`cleave_retain`/
/// `cleave_release` below are Phase 0 of it (the always-on, no-static-
/// analysis-needed reference-counting fallback, correct on its own before
/// any region/pool specialization exists) — not yet wired into `mlir_lower.
/// rs`'s own struct/tensor construction, so this function itself still
/// leaks unconditionally for now.
#[unsafe(no_mangle)]
pub extern "C" fn cleave_alloc(size: i64) -> *mut u8 {
    let layout = std::alloc::Layout::from_size_align(size as usize, 16).expect("cleave_alloc: invalid layout");
    unsafe { std::alloc::alloc(layout) }
}

/// The header `cleave_alloc_rc` prepends to every allocation it makes —
/// `refcount` first (what `cleave_retain`/`cleave_release` touch on every
/// call, so it wants to be at a fixed, zero offset from the header's own
/// base rather than computed from `data_size`), `data_size` second (needed
/// only once, at the final `cleave_release` that actually frees, to
/// reconstruct the exact `Layout` `std::alloc::dealloc` requires — Rust's
/// own allocator API has no "figure out my own layout" query, unlike a
/// libc-style `malloc`/`free` pair). 16 bytes total, matching `cleave_
/// alloc`'s own existing 16-byte alignment (`repr(C)` to fix the field
/// order/no-padding layout `RC_HEADER_SIZE`'s own arithmetic below assumes —
/// Rust's default struct layout is otherwise free to reorder fields).
#[repr(C)]
struct RcHeader {
    refcount: i64,
    data_size: i64,
}

const RC_HEADER_SIZE: usize = std::mem::size_of::<RcHeader>();

/// Read `ptr`'s own header — every `cleave_alloc_rc`-returned pointer sits
/// exactly `RC_HEADER_SIZE` bytes after its own header's base, unconditionally
/// (`cleave_alloc_rc`'s own doc comment), so this offset is never optional or
/// guessed.
///
/// # Safety
/// `ptr` must be a pointer this same `cleave_alloc_rc` returned, not yet
/// freed by a `cleave_release` that reached zero — the same "only ever call
/// this on a value the matching allocator itself produced" contract every
/// other raw-pointer function in this file already carries.
unsafe fn rc_header(ptr: *mut u8) -> *mut RcHeader {
    unsafe { ptr.sub(RC_HEADER_SIZE) as *mut RcHeader }
}

/// `doc/hld.md`'s own "Memory management" section, Phase 0 — the always-on,
/// no-escape-analysis-needed reference-counting fallback (Swift ARC's own
/// "correct by itself before any elision" starting point, not a novel
/// scheme): every allocation starts with `refcount = 1` (the reference its
/// own construction site holds), `cleave_retain` on every real aliasing
/// event (a second simultaneously-live binding/field-store of the same
/// value), `cleave_release` wherever a binding's own scope ends without the
/// value escaping further — freed for real only once the count reaches
/// zero. `Ordering::Relaxed`-equivalent (plain, non-atomic reads/writes, no
/// `Atomic*` type at all) deliberately — `doc/hld.md`'s own "Threading"
/// paragraph in that section: this whole scheme is single-threaded by
/// design, the same existing assumption `pcg32_next_u32`'s own doc comment
/// below already states for this runtime's other mutable state (`cleave_
/// alloc`'s own allocator included).
///
/// A *new*, parallel primitive rather than a change to `cleave_alloc` itself
/// — not yet wired into `mlir_lower.rs`'s own construction/lowering (a
/// separate, larger step: deciding *where* retain/release calls get
/// inserted needs real CPS-level escape-analysis work, `doc/hld.md`'s own
/// still-open "exactly where retain/release operations get inserted" item)
/// — so nothing existing changes behavior by this landing.
/// **Deliberately *not* region-aware, on purpose, after a real design
/// correction** (`doc/backlog.md` — an earlier version of this function
/// *did* implicitly draw from the arena whenever a `cleave_region_enter`
/// was open anywhere in the dynamic call stack, reverted once a real
/// target case showed why that's unsound: `Optimizer::step`'s own call, in
/// `examples/mnist-interop`'s real training loop, runs *nested inside* the
/// exact same open region `net_grad`'s own call needs — for `g.2` (`net_
/// grad`'s own result) to stay valid for the whole time `Optimizer::step`
/// is reading it, the region can't close before `Optimizer::step` returns,
/// but `Optimizer::step`'s *own* newly-built `w`/`b` tensors are precisely
/// what escapes *past* that same `region_exit` (they become next
/// iteration's `net`/`state`). A single ambient "is some region open" flag
/// cannot tell these two calls' own allocation sites apart — they're both
/// live at the exact same moment. The only sound place to make that
/// distinction is per allocation *site*, at compile time, not per dynamic
/// call at runtime — matching `doc/hld.md`'s own four-operation interface
/// more literally than the reverted version did: `alloc_escaping` (this
/// function, unconditionally heap-backed, the *default* for anything not
/// individually proven local) and `alloc_local` (`cleave_alloc_local`,
/// below — explicit, opt-in, only ever emitted at a site the compiler has
/// actually proven safe) are two genuinely different entry points, not one
/// function silently branching on ambient state.
#[unsafe(no_mangle)]
pub extern "C" fn cleave_alloc_rc(data_size: i64) -> *mut u8 {
    let total = RC_HEADER_SIZE + data_size as usize;
    let layout = std::alloc::Layout::from_size_align(total, 16).expect("cleave_alloc_rc: invalid layout");
    unsafe {
        let base = std::alloc::alloc(layout);
        assert!(!base.is_null(), "cleave_alloc_rc: allocation failed");
        let header = base as *mut RcHeader;
        (*header).refcount = 1;
        (*header).data_size = data_size;
        base.add(RC_HEADER_SIZE)
    }
}

/// # Safety
/// See `rc_header`'s own safety contract — `ptr` must be a live `cleave_
/// alloc_rc` result.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cleave_retain(ptr: *mut u8) {
    unsafe {
        let header = rc_header(ptr);
        (*header).refcount += 1;
    }
}

/// Decrements `ptr`'s own refcount; once it reaches zero, actually frees the
/// whole allocation (header included) using the exact `Layout` `data_size`
/// (recorded at `cleave_alloc_rc` time) reconstructs — `std::alloc::dealloc`
/// requires the identical layout `alloc` was given, not just a matching
/// pointer. Returns whether this specific call actually freed it (refcount
/// reached zero) — `mlir_lower.rs::lower_release_cascade` needs this: a
/// struct's own cascade into its refcounted fields (a nested struct, or a
/// `#[mlir_type(tensor)]`-tagged field's own payload — neither has its own
/// separate liveness check, `store_native_shape_field`'s own doc comment)
/// is only sound *inside* the branch where the container itself was
/// genuinely destroyed, not on every call (found by direct testing, a real
/// `STATUS_HEAP_CORRUPTION`: cascading unconditionally frees a field a
/// *second*, still-live alias of the very same container still needs, the
/// moment that alias's own count merely drops from 2 to 1).
///
/// # Safety
/// See `rc_header`'s own safety contract — `ptr` must be a live `cleave_
/// alloc_rc` result, and (ordinary reference-counting discipline) this must
/// be called at most once per real reference this value's refcount was
/// actually incremented for — a redundant release below the true reference
/// count is a real use-after-free once every *counted* reference has
/// separately been released too, the same hazard any refcounting scheme has.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cleave_release(ptr: *mut u8) -> bool {
    unsafe {
        let header = rc_header(ptr);
        (*header).refcount -= 1;
        if (*header).refcount == 0 {
            // Arena-backed (`cleave_alloc_rc`'s own doc comment): never
            // individually freed here — the matching `cleave_region_exit`
            // reclaims it in bulk, along with everything else allocated
            // since. Still correctly reports `true` below either way: a
            // struct's own cascading release into its refcounted fields
            // (`mlir_lower.rs::lower_release_cascade`) must still fire
            // once *this* container's own count genuinely reaches zero,
            // regardless of which physical allocator backs it.
            if !is_in_arena(header as *mut u8) {
                let data_size = (*header).data_size;
                let total = RC_HEADER_SIZE + data_size as usize;
                let layout = std::alloc::Layout::from_size_align(total, 16)
                    .expect("cleave_release: invalid layout");
                std::alloc::dealloc(header as *mut u8, layout);
            }
            true
        } else {
            false
        }
    }
}

/// Total bytes reserved for the CPU-backend arena (`doc/hld.md`'s own
/// "Memory management" section: "one large reserved VM region... pages
/// committed lazily, exactly like an ordinary thread's own call stack").
/// A real OS-level `VirtualAlloc`-style lazy-commit reservation is the
/// eventual target (matching that section's own wording exactly) — this
/// first cut allocates the whole capacity eagerly, through the ordinary
/// system allocator, the simplest correct thing that already gives every
/// `cleave_region_enter`/`cleave_alloc_local`/`cleave_region_exit` call
/// below a real, working backing store to test against. 256 MiB: bigger
/// than any single training-loop iteration's own local footprint this
/// project's own real workload (`examples/mnist-interop`, per-sample
/// tensors well under a megabyte) plausibly needs — a real number to
/// revisit once a real workload's own peak region depth is measured, not
/// a permanent ceiling; `cleave_alloc_local`'s own overflow check exists
/// specifically so exceeding it fails loudly rather than corrupting
/// whatever memory happens to sit past the reserved region.
const ARENA_CAPACITY: usize = 256 * 1024 * 1024;

/// The arena's own base address, lazily allocated on first use —
/// `AtomicUsize` (an address, not a `*mut u8`) purely so this can be a
/// `static` at all (raw pointers aren't `Sync`) — the same reasoning
/// `PCG_STATE`'s own doc comment gives for using an atomic type here
/// despite this runtime being single-threaded by design throughout
/// (`doc/hld.md`'s own "Threading" paragraph, which explicitly names this
/// exact region/pool/refcount scheme as staying non-atomic by design):
/// `Ordering::Relaxed` everywhere below, no real concurrency, just a
/// `Sync`-satisfying container for otherwise-plain mutable state.
static ARENA_BASE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
/// Byte offset of the arena's own current bump cursor, relative to
/// `ARENA_BASE` — `cleave_region_enter`/`cleave_alloc_local`/`cleave_
/// region_exit` below are the only three operations that ever touch it.
static ARENA_CURSOR: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
/// How many `cleave_region_enter` calls are currently open, without a
/// matching `cleave_region_exit` yet. **Not** consulted by `cleave_alloc_
/// rc` (that function's own doc comment has the real design reasoning why
/// not) — this exists purely so `cleave_alloc_local` can debug-assert it's
/// never emitted at a site with no region actually open, a real compiler-
/// bug detector, not a runtime branch point. A *count*, not a bool,
/// because regions nest (`rc_tests`'s own nesting coverage, below) — an
/// inner `region_exit`, with an outer region still open, must leave this
/// above zero.
static REGION_DEPTH: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Returns the arena's own base address, allocating the whole reserved
/// region on the very first call (from whichever of the three arena
/// functions below runs first — deliberately not tied to process startup,
/// so a program that never uses the arena at all never pays for it).
fn arena_base() -> *mut u8 {
    use std::sync::atomic::Ordering::Relaxed;
    let existing = ARENA_BASE.load(Relaxed);
    if existing != 0 {
        return existing as *mut u8;
    }
    let layout =
        std::alloc::Layout::from_size_align(ARENA_CAPACITY, 64).expect("cleave_region_enter: invalid arena layout");
    let base = unsafe { std::alloc::alloc(layout) };
    assert!(!base.is_null(), "cleave_region_enter: arena allocation failed");
    ARENA_BASE.store(base as usize, Relaxed);
    base
}

/// Bumps `size` bytes off the arena's own cursor, 16-byte aligned — matches
/// `cleave_alloc_rc`'s own `Layout::from_size_align(total, 16)` exactly
/// (not this project's own 64-byte vectorization width: checked directly,
/// the disassembled tensor-payload accesses reached through `cleave_
/// alloc_rc` already use the *unaligned* masked-load/store forms, `vmovups
/// `/`llvm.intr.masked.load`, not `vmovaps` — 64-byte alignment was never
/// actually assumed for this allocator's own output, only for the
/// unrelated, separately-`alignment = 64`-tagged `memref.alloc()` locals
/// `mlir_lower.rs::alloc_llvm_value` builds) — bumps the cursor forward,
/// returns the pre-bump address. The one real bump-allocation primitive:
/// `cleave_alloc_local` (`doc/hld.md`'s own named entry point) is this
/// function's only caller.
fn arena_bump(size: usize) -> *mut u8 {
    use std::sync::atomic::Ordering::Relaxed;
    let base = arena_base();
    let cursor = ARENA_CURSOR.load(Relaxed);
    let aligned = (cursor + 15) & !15;
    let new_cursor = aligned + size;
    assert!(
        new_cursor <= ARENA_CAPACITY,
        "cleave arena exhausted ({new_cursor} > {ARENA_CAPACITY} bytes) -- \
         a real overflow path (grow, or fall back to the ordinary allocator) is not built yet"
    );
    ARENA_CURSOR.store(new_cursor, Relaxed);
    unsafe { base.add(aligned) }
}

/// Whether `ptr` falls inside the arena's own reserved address range —
/// `cleave_release`'s own arena-vs-heap decision (its own doc comment).
/// `ARENA_BASE == 0` (the arena has never been used at all in this
/// process) short-circuits to `false` directly, rather than comparing
/// against a base of `0` — a real heap pointer is never `0`, but relying
/// on that coincidence instead of checking explicitly would be the kind
/// of "probably fine" this codebase's own established discipline avoids.
fn is_in_arena(ptr: *mut u8) -> bool {
    use std::sync::atomic::Ordering::Relaxed;
    let base = ARENA_BASE.load(Relaxed);
    if base == 0 {
        return false;
    }
    let addr = ptr as usize;
    addr >= base && addr < base + ARENA_CAPACITY
}

/// `doc/hld.md`'s own `region_enter(size) -> handle` — the CPU backend for
/// it ("`region_enter`/`region_exit` as pointer arithmetic... exactly like
/// an ordinary thread's own call stack", same section). `size` is accepted
/// (matching that interface's own signature — a future caller may want to
/// pre-validate against it) but not consumed by one eager bump here:
/// `cleave_alloc_local` below does the actual per-value bumping, each call
/// already knowing its own exact size, and nothing between `region_enter`/
/// `region_exit` needs `size` for anything else yet. The *handle* returned
/// is simply the cursor's own value at entry — `cleave_region_exit`
/// rewinds straight back to it, discarding everything allocated since,
/// unconditionally, matching the region scheme's own "provably dead the
/// instant the tail call fires" premise (`doc/hld.md`, same section):
/// nothing is meant to call `cleave_region_enter` for a value the compiler
/// hasn't already proven doesn't escape past the matching `region_exit`.
/// A plain `i64` offset, not a pointer — an arena that later grows (or
/// moves) can still honor an old handle; a raw `*mut u8` captured before a
/// hypothetical reallocation couldn't.
#[unsafe(no_mangle)]
pub extern "C" fn cleave_region_enter(_size: i64) -> i64 {
    use std::sync::atomic::Ordering::Relaxed;
    arena_base();
    REGION_DEPTH.fetch_add(1, Relaxed);
    ARENA_CURSOR.load(Relaxed) as i64
}

/// `doc/hld.md`'s own `alloc_local(handle, size) -> ptr` — carves `size`
/// bytes out of the arena at the current cursor (16-byte aligned — see
/// `arena_bump`'s own doc comment for why 16, not this project's own
/// 64-byte vectorization width), bumps the cursor forward. `handle` (the region this
/// allocation conceptually belongs to) isn't itself read here —
/// correctness only needs the matching `cleave_region_exit` to eventually
/// rewind past it, not a per-allocation check against it (nesting is a
/// strict stack discipline by construction, enforced by `region_enter`/
/// `region_exit` call *pairing*, not by this function auditing individual
/// allocations against their own handle).
///
/// **Writes the exact same `RcHeader` `cleave_alloc_rc` does, at the same
/// relative offset** — not a separate, lighter-weight shape. The compiler
/// picks `cleave_alloc_rc` vs `cleave_alloc_local` once, per allocation
/// *site*, at compile time (`cleave_alloc_rc`'s own doc comment); every
/// `cleave_retain`/`cleave_release` call downstream is emitted by the
/// *same* codegen either way, with no idea which allocator actually backed
/// the value it's touching — so both must produce an identical header, or
/// retain/release would read garbage off a bare arena allocation with no
/// header at all. `size` is `data_size` alone, matching `cleave_alloc_rc`'s
/// own parameter convention exactly — the header's own extra bytes are
/// accounted for here, not by the caller.
///
/// `REGION_DEPTH == 0` at this call is *always* a genuine compiler bug
/// (this function must only ever be emitted at a site already inside a
/// matching `region_enter`/`region_exit` pair) — `assert_region_open`
/// (right below) catches it loudly and unconditionally (not gated behind
/// `debug_assertions` — this project's own established convention is
/// testing under `cargo test --release`, which disables it by default; a
/// check that only exists in debug builds would never actually run under
/// that workflow), the same posture `cleave_alloc_rc`'s own `assert!(!
/// base.is_null(), ...)` already takes on its own always-on allocation-
/// failure check. A separate, plain (not `extern "C"`) function rather
/// than an inline `assert!` here, purely so this crate's own tests can
/// `catch_unwind` it directly: a panic *inside* an `extern "C"` function
/// cannot unwind at all (confirmed directly — Rust aborts the whole
/// process instead, `panic_cannot_unwind`, not something `catch_unwind`
/// can observe), so the only way to test this check's own panic behavior
/// is to keep it in an ordinary Rust function `cleave_alloc_local` merely
/// calls into.
fn assert_region_open() {
    use std::sync::atomic::Ordering::Relaxed;
    assert!(
        REGION_DEPTH.load(Relaxed) > 0,
        "cleave_alloc_local called with no region open -- a real compiler bug, \
         never a legitimate runtime condition"
    );
}
#[unsafe(no_mangle)]
pub extern "C" fn cleave_alloc_local(_handle: i64, size: i64) -> *mut u8 {
    assert_region_open();
    let total = RC_HEADER_SIZE + size as usize;
    let base = arena_bump(total);
    unsafe {
        let header = base as *mut RcHeader;
        (*header).refcount = 1;
        (*header).data_size = size;
        base.add(RC_HEADER_SIZE)
    }
}

/// `doc/hld.md`'s own `region_exit(handle)` — "a pointer rewind, nothing
/// more" (that section's own words, for the CPU backend specifically):
/// every byte allocated since the matching `region_enter` becomes
/// available for reuse, unconditionally, no per-object bookkeeping, no
/// `cleave_release` calls needed for anything that lived purely in this
/// region.
#[unsafe(no_mangle)]
pub extern "C" fn cleave_region_exit(handle: i64) {
    use std::sync::atomic::Ordering::Relaxed;
    ARENA_CURSOR.store(handle as usize, Relaxed);
    REGION_DEPTH.fetch_sub(1, Relaxed);
}

/// `cleave_release`'s own `bool` result ("did this call actually free the
/// block"), discarded — matches `free`'s own `(ptr) -> ()` C signature
/// exactly. Exists purely so `unify_alloc.rs`'s own `llvm.call @free` ->
/// `llvm.call @cleave_release_void` rewrite can be a **plain callee-symbol
/// rename**, nothing else: melior's own `remove_from_parent` is confirmed
/// unsafe to call at all on real ops from this pipeline (`dps_rewrite.rs`'s
/// own doc comment on `memcpy`, and — checked again here, since a *
/// different* op kind isn't automatically covered by that same finding —
/// on `memref.dealloc`/`memref.alloc` too: erasing either one succeeds at
/// the call site itself but corrupts internal state that only crashes
/// later, at module teardown), so *rebuilding* a call op with a different
/// result arity (`free`'s `()` vs `cleave_release`'s own `i1`) is exactly
/// the kind of erase-and-replace this project's own established discipline
/// avoids wherever a same-shape alternative exists instead. `llvm.call
/// @malloc(size) -> ptr` already matches `cleave_alloc_rc`'s own real
/// signature byte-for-byte, needing no such wrapper at all — this one
/// exists only because `free`'s own C signature returns nothing.
///
/// # Safety
/// See `rc_header`'s own safety contract — `ptr` must be a live `cleave_
/// alloc_rc` result.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cleave_release_void(ptr: *mut u8) {
    unsafe {
        cleave_release(ptr);
    }
}

/// Reads `ptr`'s own current refcount without changing it — a real,
/// necessary observation point for tests (see `rc_tests` below); not part
/// of the "real" `extern fn` surface `mlir_lower.rs`-generated code ever
/// calls, so no `#[unsafe(no_mangle)]`/`extern "C"` needed.
///
/// # Safety
/// See `rc_header`'s own safety contract.
#[cfg(test)]
unsafe fn rc_count(ptr: *mut u8) -> i64 {
    unsafe { (*rc_header(ptr)).refcount }
}

#[cfg(test)]
mod rc_tests {
    use super::*;

    /// One test, deliberately, covering `cleave_alloc_rc`/`cleave_retain`/
    /// `cleave_release`/`cleave_release_void` *and* the arena (`cleave_
    /// region_enter`/`cleave_alloc_local`/`cleave_region_exit`) together —
    /// the same reasoning `rand_tests::pcg32_behaves_correctly` already
    /// gives for consolidating everything touching one piece of shared
    /// global state into a single test: `REGION_DEPTH`/`ARENA_CURSOR` are
    /// process-wide, and splitting the arena-touching checks into separate
    /// `#[test]` fns would let one test's own open region race another's
    /// `cleave_alloc_local` call (or the "no region open" check just
    /// below) into seeing state left behind by a *different*, concurrently
    /// running test — an intermittent failure with no bug in either
    /// mechanism. `cleave_alloc_rc` itself no longer touches this state at
    /// all (`cleave_alloc_rc`'s own doc comment — a real design correction,
    /// not the original plan) — kept consolidated anyway, since it's
    /// simplest to keep every test that touches *any* of this file's own
    /// shared global mutable state (PCG state excepted, already its own
    /// separate consolidated test) in one place, not because it strictly
    /// needs to be any more.
    #[test]
    fn refcounting_and_the_arena_behave_correctly() {
        // `cleave_alloc_local` outside any open region is a real compiler
        // bug (its own doc comment) — the assertion exists to catch it
        // loudly, unconditionally (not debug-only — this project's own
        // tests run under `--release`). Checked here, first, before this
        // same test opens any region of its own, so `REGION_DEPTH` is
        // genuinely `0` at this point (no other test touches this state
        // concurrently — see this test's own doc comment above for why
        // that matters).
        {
            let result = std::panic::catch_unwind(assert_region_open);
            assert!(result.is_err(), "assert_region_open with no open region should panic");
        }

        unsafe {
            // -- Ordinary (no region open) `cleave_alloc_rc` --
            let ptr = cleave_alloc_rc(8);
            assert_eq!(rc_count(ptr), 1);
            assert!(
                !is_in_arena(ptr),
                "no region is open here -- this must be an ordinary heap allocation"
            );
            cleave_release(ptr);

            let ptr = cleave_alloc_rc(8);
            cleave_retain(ptr);
            assert_eq!(rc_count(ptr), 2);
            cleave_release(ptr);
            assert_eq!(rc_count(ptr), 1);
            cleave_release(ptr);

            // Real proof this isn't just header bookkeeping — the returned
            // pointer is a real, correctly-offset, correctly-sized data
            // region, not just something that satisfies the refcount
            // checks alone.
            let ptr = cleave_alloc_rc(8) as *mut i64;
            *ptr = 0x1234_5678_9abc_def0;
            assert_eq!(*ptr, 0x1234_5678_9abc_def0);
            cleave_release(ptr as *mut u8);

            // A "release-to-zero actually calls dealloc, not just zeroes
            // the count" check was tried here and removed, not left red:
            // it asserted a fresh allocation reuses the just-freed
            // address, found directly to be unreliable against the real
            // system allocator (Windows' own allocator doesn't guarantee
            // immediate reuse the way some allocators' fast paths do) — a
            // real, non-deterministic property this test can't black-box
            // verify without a custom `#[global_allocator]` tracking
            // wrapper, not worth building for this one check.

            let a = cleave_alloc_rc(8);
            let b = cleave_alloc_rc(8);
            cleave_retain(a);
            assert_eq!(rc_count(a), 2);
            assert_eq!(rc_count(b), 1);
            cleave_release(a);
            cleave_release(a);
            cleave_release(b);

            let ptr = cleave_alloc_rc(8) as *mut i64;
            *ptr = 42;
            cleave_retain(ptr as *mut u8);
            assert_eq!(rc_count(ptr as *mut u8), 2);
            cleave_release_void(ptr as *mut u8);
            assert_eq!(
                rc_count(ptr as *mut u8),
                1,
                "one release_void call should drop the count by exactly one, same as release"
            );
            // Second (final) release through the real `cleave_release` --
            // confirms `release_void`'s own first call genuinely shares the
            // same header/count `cleave_release` itself uses, not a
            // separate bookkeeping path.
            assert!(
                cleave_release(ptr as *mut u8),
                "the final release should report that it actually freed the block"
            );
        }

        // -- The arena itself, `cleave_region_enter`/`cleave_alloc_local`/
        // `cleave_region_exit` --

        // A single allocation is real, writable memory of the requested
        // size — not just an address that satisfies bookkeeping alone.
        let h0 = cleave_region_enter(64);
        let p0 = cleave_alloc_local(h0, 64) as *mut i64;
        unsafe {
            *p0 = 0x1234_5678_9abc_def0;
            assert_eq!(*p0, 0x1234_5678_9abc_def0);
        }
        cleave_region_exit(h0);

        // Two allocations in the same region land at different,
        // non-overlapping addresses.
        let h1 = cleave_region_enter(256);
        let a = cleave_alloc_local(h1, 64);
        let b = cleave_alloc_local(h1, 64);
        assert_ne!(a, b, "two live allocations must not overlap");
        unsafe {
            std::ptr::write_bytes(a, 0xaa, 64);
            std::ptr::write_bytes(b, 0xbb, 64);
            assert_eq!(*a, 0xaa, "writing through `b` must not have touched `a`");
            assert_eq!(*b, 0xbb);
        }
        cleave_region_exit(h1);

        // `region_exit` really rewinds — a fresh allocation right after
        // reuses the exact address just freed (the bump cursor moved
        // back, not forward past it).
        let h2 = cleave_region_enter(64);
        let reused = cleave_alloc_local(h2, 64);
        assert_eq!(reused, a, "region_exit should have rewound the cursor back to `a`'s own address");
        cleave_region_exit(h2);

        // Nesting: entering a second region *inside* a still-open one,
        // exiting the inner one, must leave the outer region's own
        // already-live allocation completely untouched.
        let outer = cleave_region_enter(128);
        let outer_ptr = cleave_alloc_local(outer, 64) as *mut i64;
        unsafe {
            *outer_ptr = 111;
        }
        let inner = cleave_region_enter(64);
        let inner_ptr = cleave_alloc_local(inner, 64) as *mut i64;
        unsafe {
            *inner_ptr = 222;
        }
        cleave_region_exit(inner);
        unsafe {
            assert_eq!(*outer_ptr, 111, "exiting the nested inner region corrupted the outer one's own live data");
        }
        cleave_region_exit(outer);

        // 16-byte alignment, unconditionally (`arena_bump`'s own doc
        // comment has the real reasoning for 16, not 64).
        let h3 = cleave_region_enter(256);
        let p = cleave_alloc_local(h3, 17) as usize; // an odd size on purpose
        assert_eq!(p % 16, 0, "alloc_local's own result must be 16-byte aligned");
        cleave_region_exit(h3);

        // -- `cleave_alloc_local`, refcount-header-compatible --
        // `cleave_alloc_rc` is deliberately *not* region-aware any more
        // (its own doc comment has the real design correction) -- a
        // program that never opens a region gets ordinary heap allocation
        // throughout, unconditionally.
        unsafe {
            let heap_ptr = cleave_alloc_rc(8);
            assert!(!is_in_arena(heap_ptr), "no region open -- cleave_alloc_rc must stay heap-backed");
            cleave_release(heap_ptr);
        }

        // `cleave_alloc_local` writes the *exact same* header shape --
        // real, correctly-offset `refcount`/`data_size` fields, not just
        // raw bump-allocated bytes -- so `cleave_retain`/`cleave_release`
        // (emitted identically by the same codegen regardless of which
        // allocator actually backed a given value) work on it exactly as
        // they would on a `cleave_alloc_rc` result.
        unsafe {
            let h = cleave_region_enter(64);
            let ptr = cleave_alloc_local(h, 8);
            assert!(is_in_arena(ptr), "should be arena-backed inside an open region");
            assert_eq!(rc_count(ptr), 1, "cleave_alloc_local must write a real, correct RcHeader");
            cleave_retain(ptr);
            assert_eq!(rc_count(ptr), 2);
            // Dropping back to a live count of 1 must *not* attempt to
            // free anything (arena-backed) -- if it wrongly tried to
            // `dealloc` arena memory, this would crash outright.
            assert!(!cleave_release(ptr), "should not report freed while still retained");
            // Reaching zero on an arena-backed allocation reports `true`
            // (needed for cascading release, `cleave_release`'s own doc
            // comment) but must not crash by trying to `dealloc` memory
            // that was never individually `alloc`'d.
            assert!(
                cleave_release(ptr),
                "reaching zero must still report true even when arena-backed, for cascading release"
            );
            cleave_region_exit(h);
        }
    }

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
