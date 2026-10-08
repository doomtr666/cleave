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

pub mod checkpoint;

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
    let p = unsafe { std::alloc::alloc(layout) };
    if *alloc_stats::ENABLED {
        alloc_stats::record(size as usize, true, Some((p as usize, size as usize)));
    }
    p
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
/// Segregated free-list cache behind `cleave_alloc_rc`/`cleave_release`'s
/// own non-arena path — **the simpler replacement for a much more involved
/// design that was sketched but never built** (`doc/backlog.md`'s own
/// former "depth-bounded pool per loop-carried allocation site" entry): that
/// version needed a brand-new CPS-level static classification (which
/// top-level allocation sites are loop-carried, single-call-site,
/// statically-fixed-size) before a single line of `cleave-rt` code could
/// even be reached. The real target it was chasing — `Optimizer::step`'s
/// own 16 tensor leaves, replaced every training-loop iteration, retired
/// almost immediately after (`region_analysis.rs`'s own module doc comment
/// has the full story) — never actually needed to know *which* call site
/// produced a given allocation, only that **the same handful of sizes
/// recur every iteration**. Bucketing by size class alone captures that
/// directly, generalizes to every `cleave_alloc_rc` caller in the program
/// (not just the one motivating loop), and needs no new compiler analysis
/// at all: "pay `RtlAllocateHeap`/`RtlFreeHeap` for a given size at most
/// once, ever, for the rest of the process's life" is what a segregated
/// free list *is*, not something bolted onto it.
///
/// **Real, found-by-testing correction to this section's own original
/// claim** ("`OMP_NUM_THREADS` parallelism lives entirely inside MLIR-
/// generated compute loops, never reaching `cleave-rt`") — checked directly
/// against the real `mnist-interop` kernel's own disassembled `.o`
/// (`llvm-objdump`, this project's own established methodology): `cleave_
/// alloc_rc`/`cleave_release` calls exist *inside* several `..omp_par.N`
/// outlined parallel-region bodies (per-thread scratch tensors for the
/// tiled matmul), genuinely reachable from multiple OpenMP worker threads
/// at once whenever `OMP_NUM_THREADS > 1`. `cleave_retain`/`cleave_
/// release`'s own plain (non-atomic) refcount increment/decrement stay
/// sound regardless — each such scratch tensor's own header is thread-
/// private, no two threads ever touch the *same* header — but `FREE_LISTS`
/// itself is genuinely shared, global, mutable state every thread reaches
/// through the *identical* allocator entry points, with nothing here ever
/// synchronizing it: a real, reproducible race (two threads racing `cleave_
/// alloc_rc`'s own free-list pop, both reading the same head before either
/// writes the new one back, handing the *same* block out twice at once).
/// `POOL_LOCK` below is the fix — a spinlock, not a full OS mutex: the
/// critical section is a handful of pointer reads/writes, cheap enough
/// that spinning beats a syscall-backed lock's own overhead, and the
/// refcount increment/decrement themselves stay outside it (still
/// per-header-private, no need to serialize those too).
static POOL_LOCK: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

// `CLEAVE_DEBUG_POOL=1` -- the pool checks every block it hands out and takes
// back against the set of freed ones (`PARKED`): a retain or release of a
// freed block, or a block freed twice, stops the program at once, naming the
// block, its allocation and the offending call's source line. Slow (a global
// set touched on every allocation and release), for finding a refcounting
// error; `tests/examples.rs` runs every example under it.
static CLEAVE_DEBUG_POOL: std::sync::LazyLock<bool> =
    std::sync::LazyLock::new(|| std::env::var("CLEAVE_DEBUG_POOL").is_ok());
static PARKED: std::sync::Mutex<Option<std::collections::HashSet<usize>>> =
    std::sync::Mutex::new(None);
fn parked_insert(base: usize) -> bool {
    PARKED.lock().unwrap().get_or_insert_with(Default::default).insert(base)
}
fn parked_remove(base: usize) -> bool {
    PARKED.lock().unwrap().get_or_insert_with(Default::default).remove(&base)
}
fn parked_contains(base: usize) -> bool {
    PARKED.lock().unwrap().get_or_insert_with(Default::default).contains(&base)
}

// `CLEAVE_TRACE_RC=1` -- a ledger of every allocation, retain and release
// (block, allocation serial, resulting count, cleave source line), for a
// small reproduction: one line per call floods a real workload
// (`CLEAVE_TRACE_SIZE` narrows it to one size).
static CLEAVE_TRACE_RC: std::sync::LazyLock<bool> =
    std::sync::LazyLock::new(|| std::env::var("CLEAVE_TRACE_RC").is_ok());

// `CLEAVE_TRACE_SIZE=<bytes>` -- a complete allocate/retain/release ledger
// for blocks of exactly one `data_size`, and nothing else. `CLEAVE_TRACE_RC`
// above is the same idea whole-program, and its own doc comment already
// admits it floods any real workload; narrowing it to the single size class
// a crash message has already named is what makes it usable on the *real*
// `mnist-interop` kernel rather than only on a minimal repro. That is
// exactly how `doc/plan-region-arena.md`'s own `data_size=104` double-
// release was root-caused (§7): the fatal message named the size, this flag
// then produced every event for that size and nothing else.
static CLEAVE_TRACE_SIZE: std::sync::LazyLock<Option<i64>> = std::sync::LazyLock::new(|| {
    std::env::var("CLEAVE_TRACE_SIZE")
        .ok()
        .and_then(|v| v.parse::<i64>().ok())
});

/// Whether any of the checks and traces above is on: one test on
/// `cleave_release`'s path instead of one per switch.
static DIAGNOSTICS: std::sync::LazyLock<bool> =
    std::sync::LazyLock::new(|| *CLEAVE_DEBUG_POOL || *CLEAVE_TRACE_RC || CLEAVE_TRACE_SIZE.is_some());

// `CLEAVE_ALLOC_STATS=1` -- every allocation counted, by call site, printed
// to stderr at exit: how many, how many bytes, and how many missed the pool
// (`fresh`: new memory from the system, whose first touch page-faults). The
// measure of what a program materializes, deterministic where a timing
// isn't: a rewrite that fuses two loops or writes into an existing buffer
// shows up here as fewer bytes, run after run. Per step, by difference: two
// runs of a training loop that differ only in their number of steps.
// A site is the stack above the allocator, captured raw (a few
// microseconds) and symbolized only at exit.
//
// It also follows what is alive: each block from its allocation to its
// free, the bytes asked for and the bytes its size class takes, and the
// freed blocks parked in the pool (`pool_push`), never given back to the
// system. At the peak of the live bytes, the live bytes by site are
// recorded, and printed at exit with the process's own peak working set:
// what a step holds at its high point, and where it was allocated.
mod alloc_stats {
    use std::collections::HashMap;
    use std::sync::{LazyLock, Mutex, Once};

    pub static ENABLED: LazyLock<bool> = LazyLock::new(|| std::env::var("CLEAVE_ALLOC_STATS").is_ok());

    const FRAMES: usize = 8;

    #[derive(Default)]
    struct Site {
        count: u64,
        bytes: u64,
        fresh: u64,
        fresh_bytes: u64,
    }

    /// Keyed by the stack and the size: one site allocating several shapes
    /// (a synthesized gradient, attributed to its one `grad` line) reads
    /// as one row per shape.
    static SITES: Mutex<Option<HashMap<([usize; FRAMES], usize), Site>>> = Mutex::new(None);
    static REPORT_AT_EXIT: Once = Once::new();

    type SiteKey = ([usize; FRAMES], usize);

    /// What is alive, under one lock: each live block's site and sizes, by
    /// address, and the totals.
    #[derive(Default)]
    struct Live {
        blocks: HashMap<usize, (SiteKey, u64, u64)>,
        by_site: HashMap<SiteKey, (u64, u64)>,
        bytes: u64,
        class_bytes: u64,
        parked: u64,
        peak_bytes: u64,
        peak_class_bytes: u64,
        peak_parked: u64,
        peak_held: u64,
        at_peak: Vec<(SiteKey, (u64, u64))>,
        at_peak_parked: u64,
        at_peak_class_bytes: u64,
        at_peak_bytes: u64,
    }

    static LIVE: Mutex<Option<Live>> = Mutex::new(None);

    unsafe extern "C" {
        fn atexit(callback: extern "C" fn()) -> i32;
    }

    /// An allocation of `bytes`; `block` (its base) and the bytes its size
    /// class takes when it is freed individually later (`forget`), `None`
    /// for an arena or pool allocation, not followed.
    pub fn record(bytes: usize, fresh: bool, block: Option<(usize, usize)>) {
        REPORT_AT_EXIT.call_once(|| unsafe {
            atexit(report);
        });
        let mut ips = [0usize; FRAMES];
        let mut n = 0;
        backtrace::trace(|frame| {
            ips[n] = frame.ip() as usize;
            n += 1;
            n < FRAMES
        });
        {
            let mut sites = SITES.lock().unwrap();
            let site = sites.get_or_insert_with(Default::default).entry((ips, bytes)).or_default();
            site.count += 1;
            site.bytes += bytes as u64;
            if fresh {
                site.fresh += 1;
                site.fresh_bytes += bytes as u64;
            }
        }
        let Some((base, class_bytes)) = block else { return };
        let key = (ips, bytes);
        let (bytes, class_bytes) = (bytes as u64, class_bytes as u64);
        let mut live = LIVE.lock().unwrap();
        let live = live.get_or_insert_with(Default::default);
        live.blocks.insert(base, (key, bytes, class_bytes));
        let site = live.by_site.entry(key).or_default();
        site.0 += bytes;
        site.1 += class_bytes;
        live.bytes += bytes;
        live.class_bytes += class_bytes;
        if !fresh {
            live.parked = live.parked.saturating_sub(class_bytes);
        }
        live.peak_class_bytes = live.peak_class_bytes.max(live.class_bytes);
        live.peak_held = live.peak_held.max(live.class_bytes + live.parked);
        if live.bytes > live.peak_bytes {
            // Recorded again only 1% above the last record: a snapshot copies
            // every site, not to be taken on each allocation of a ramp.
            if live.at_peak.is_empty() || live.bytes > live.at_peak_bytes + live.at_peak_bytes / 100 {
                live.at_peak = live.by_site.iter().filter(|(_, v)| v.0 > 0).map(|(k, v)| (*k, *v)).collect();
                live.at_peak_parked = live.parked;
                live.at_peak_class_bytes = live.class_bytes;
                live.at_peak_bytes = live.bytes;
            }
            live.peak_bytes = live.bytes;
        }
    }

    /// The block at `base` freed; `parked` when it goes to the pool rather
    /// than back to the system.
    pub fn forget(base: usize, parked: bool) {
        let mut live = LIVE.lock().unwrap();
        let Some(live) = live.as_mut() else { return };
        let Some((key, bytes, class_bytes)) = live.blocks.remove(&base) else { return };
        if let Some(site) = live.by_site.get_mut(&key) {
            site.0 -= bytes;
            site.1 -= class_bytes;
        }
        live.bytes -= bytes;
        live.class_bytes -= class_bytes;
        if parked {
            live.parked += class_bytes;
            live.peak_parked = live.peak_parked.max(live.parked);
            live.peak_held = live.peak_held.max(live.class_bytes + live.parked);
        }
    }

    #[repr(C)]
    #[derive(Default)]
    struct ProcessMemoryCounters {
        cb: u32,
        page_fault_count: u32,
        peak_working_set_size: usize,
        working_set_size: usize,
        quota_peak_paged_pool_usage: usize,
        quota_paged_pool_usage: usize,
        quota_peak_non_paged_pool_usage: usize,
        quota_non_paged_pool_usage: usize,
        pagefile_usage: usize,
        peak_pagefile_usage: usize,
    }

    #[cfg(windows)]
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentProcess() -> isize;
        fn K32GetProcessMemoryInfo(process: isize, counters: *mut ProcessMemoryCounters, cb: u32) -> i32;
    }

    /// The process's peak working set and peak private bytes, when the
    /// system tells them.
    fn process_peaks() -> Option<(u64, u64)> {
        #[cfg(windows)]
        unsafe {
            let mut c = ProcessMemoryCounters {
                cb: std::mem::size_of::<ProcessMemoryCounters>() as u32,
                ..Default::default()
            };
            if K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) != 0 {
                return Some((c.peak_working_set_size as u64, c.peak_pagefile_usage as u64));
            }
        }
        None
    }

    fn report_peak(live: &Live, labels: &mut HashMap<[usize; FRAMES], String>) {
        let mb = |b: u64| b as f64 / (1 << 20) as f64;
        eprintln!(
            "CLEAVE_ALLOC_STATS peak: live {:.1} MiB asked, {:.1} MiB in size classes; \
             parked in the pool {:.1} MiB at most; held (live in classes + parked) {:.1} MiB at most",
            mb(live.peak_bytes),
            mb(live.peak_class_bytes),
            mb(live.peak_parked),
            mb(live.peak_held),
        );
        if let Some((working_set, private)) = process_peaks() {
            eprintln!(
                "CLEAVE_ALLOC_STATS process: peak working set {:.1} MiB, peak private {:.1} MiB",
                mb(working_set),
                mb(private)
            );
        }
        let (slices, committed) = super::arena_commitment();
        eprintln!(
            "CLEAVE_ALLOC_STATS arenas: {slices} threads' slices, {:.1} MiB committed",
            mb(committed as u64)
        );
        let mut by_label: HashMap<(String, usize), (u64, u64, u64)> = HashMap::new();
        for ((ips, bytes), (asked, class)) in &live.at_peak {
            let label = labels.entry(*ips).or_insert_with(|| label(ips)).clone();
            let e = by_label.entry((label, *bytes)).or_default();
            e.0 += asked / (*bytes).max(1) as u64;
            e.1 += asked;
            e.2 += class;
        }
        let mut rows: Vec<_> = by_label.into_iter().collect();
        rows.sort_by(|a, b| b.1.1.cmp(&a.1.1));
        eprintln!(
            "live at the peak, by site ({:.1} MiB asked, {:.1} MiB in size classes, {:.1} MiB parked then):",
            mb(live.at_peak_bytes),
            mb(live.at_peak_class_bytes),
            mb(live.at_peak_parked)
        );
        eprintln!("{:>8} {:>11} {:>11} {:>12}  site", "blocks", "MiB", "class MiB", "bytes each");
        for ((label, bytes), (count, asked, class)) in rows.iter().take(40) {
            eprintln!("{count:>8} {:>11.1} {:>11.1} {bytes:>12}  {label}", mb(*asked), mb(*class));
        }
        eprintln!();
    }

    /// The first frame above the runtime: the generated function that
    /// allocated, with its `.cleave` line when there is debug info.
    fn label(ips: &[usize; FRAMES]) -> String {
        for &ip in ips.iter().filter(|&&ip| ip != 0) {
            let mut found: Option<String> = None;
            backtrace::resolve(ip as *mut std::ffi::c_void, |symbol| {
                if found.is_some() {
                    return;
                }
                let name = symbol.name().map(|n| n.to_string()).unwrap_or_default();
                if name.is_empty()
                    || name.contains("cleave_rt::")
                    || name.contains("backtrace::")
                    || name.starts_with("cleave_alloc")
                {
                    return;
                }
                let line = match (symbol.filename(), symbol.lineno()) {
                    (Some(file), Some(line)) => {
                        let file = file.to_string_lossy();
                        format!(" @ {}:{line}", file.rsplit(['/', '\\']).next().unwrap_or(&file))
                    }
                    _ => String::new(),
                };
                found = Some(format!("{name}{line}"));
            });
            if let Some(found) = found {
                return found;
            }
        }
        "<unknown>".to_string()
    }

    extern "C" fn report() {
        let Some(sites) = SITES.lock().unwrap().take() else { return };
        let mut labels: HashMap<[usize; FRAMES], String> = HashMap::new();
        if let Some(live) = LIVE.lock().unwrap().take() {
            report_peak(&live, &mut labels);
        }
        let mut by_label: HashMap<(String, usize), Site> = HashMap::new();
        for ((ips, bytes), site) in &sites {
            let label = labels.entry(*ips).or_insert_with(|| label(ips)).clone();
            let entry = by_label.entry((label, *bytes)).or_default();
            entry.count += site.count;
            entry.bytes += site.bytes;
            entry.fresh += site.fresh;
            entry.fresh_bytes += site.fresh_bytes;
        }
        let mut rows: Vec<_> = by_label.into_iter().collect();
        rows.sort_by(|a, b| b.1.bytes.cmp(&a.1.bytes));
        let total = |f: fn(&Site) -> u64| rows.iter().map(|(_, s)| f(s)).sum::<u64>();
        let mb = |b: u64| b as f64 / (1 << 20) as f64;
        eprintln!(
            "CLEAVE_ALLOC_STATS: {} allocations, {:.1} MiB; {} fresh (pool misses), {:.1} MiB",
            total(|s| s.count),
            mb(total(|s| s.bytes)),
            total(|s| s.fresh),
            mb(total(|s| s.fresh_bytes)),
        );
        eprintln!("{:>10} {:>11} {:>10} {:>8} {:>10}  site", "count", "MiB", "bytes", "fresh", "fresh MiB");
        for ((label, bytes), s) in rows.iter().take(60) {
            eprintln!(
                "{:>10} {:>11.1} {:>10} {:>8} {:>10.1}  {label}",
                s.count,
                mb(s.bytes),
                bytes,
                s.fresh,
                mb(s.fresh_bytes)
            );
        }
    }
}

// For `CLEAVE_TRACE_RC` and `CLEAVE_DEBUG_POOL`'s messages: the pool reuses a
// freed block's address for a later, unrelated allocation, which makes a
// trace keyed by pointer ambiguous -- "released twice"
// and "released once each, for two different allocations that happened to
// share an address" print identically. A monotonic serial number, assigned
// fresh on *every* `cleave_alloc_rc` call (pooled-reuse or genuinely new
// alike) and looked up (never removed) by every retain/release, tells them
// apart unambiguously.
static ALLOC_SERIAL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static ALLOC_SERIALS: std::sync::Mutex<Option<std::collections::HashMap<usize, u64>>> =
    std::sync::Mutex::new(None);
fn record_alloc_serial(base: usize) -> u64 {
    let serial = ALLOC_SERIAL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    ALLOC_SERIALS
        .lock()
        .unwrap()
        .get_or_insert_with(Default::default)
        .insert(base, serial);
    serial
}
fn current_alloc_serial(base: usize) -> Option<u64> {
    ALLOC_SERIALS
        .lock()
        .unwrap()
        .get_or_insert_with(Default::default)
        .get(&base)
        .copied()
}

/// The first backtrace frame whose own symbol file is a real `.cleave`
/// source file -- skips every Rust-side frame (`cleave_rt::cleave_retain`
/// itself, `backtrace`'s own capture machinery) to give a short, single-
/// line "who called this" label instead of a full, noisy stack dump.
/// Every `.cleave`-sourced frame on the stack, outermost-last, joined with
/// ` < ` -- `first_cleave_frame`'s own resolution logic, but collecting the
/// whole inline chain instead of stopping at the first hit. Answers "which
/// cleave function actually allocated this, and through what call path",
/// which the single-frame version can't once `--inline` has flattened
/// everything into one enclosing function (every frame then reports that
/// same outer function, and only the *line* distinguishes them). Used by
/// `CLEAVE_TRACE_SIZE`'s own ledger; see that flag's own comment.
fn cleave_frames() -> String {
    let mut frames: Vec<String> = Vec::new();
    backtrace::trace(|frame| {
        backtrace::resolve_frame(frame, |symbol| {
            if let Some(file) = symbol.filename() {
                let file = file.to_string_lossy();
                if file.ends_with(".cleave") {
                    let name = symbol
                        .name()
                        .map(|n| n.to_string())
                        .unwrap_or_else(|| "?".to_string());
                    let line = symbol.lineno().unwrap_or(0);
                    let short = file.rsplit(['/', '\\']).next().unwrap_or(&file).to_string();
                    frames.push(format!("{name}@{short}:{line}"));
                }
            }
        });
        frames.len() < 8
    });
    if frames.is_empty() {
        "<no .cleave frames>".to_string()
    } else {
        frames.join(" < ")
    }
}

fn first_cleave_frame() -> String {
    let mut cleave_label: Option<String> = None;
    // Fallback, when no frame anywhere up the stack is `.cleave`-sourced:
    // the first few *non*-`cleave_rt`/`backtrace`-internal frame names,
    // whatever they are -- still tells us something (a real symbol name,
    // even lacking file:line) rather than a bare "<unknown>".
    let mut fallback: Vec<String> = Vec::new();
    backtrace::trace(|frame| {
        backtrace::resolve_frame(frame, |symbol| {
            let name = symbol.name().map(|n| n.to_string());
            if let Some(file) = symbol.filename() {
                let file = file.to_string_lossy();
                if file.ends_with(".cleave") {
                    let name = name.clone().unwrap_or_else(|| "?".to_string());
                    let line = symbol.lineno().unwrap_or(0);
                    let short = file.rsplit(['/', '\\']).next().unwrap_or(&file);
                    cleave_label = Some(format!("{name} @ {short}:{line}"));
                }
            }
            if cleave_label.is_none() && fallback.len() < 4 {
                if let Some(n) = name {
                    if !n.contains("backtrace::") && !n.contains("cleave_rt::") {
                        fallback.push(n);
                    }
                }
            }
        });
        cleave_label.is_none() // keep walking until a `.cleave` frame is found
    });
    cleave_label.unwrap_or_else(|| format!("<no .cleave frame; stack: {}>", fallback.join(" < ")))
}

/// # Safety
/// Every `FREE_LISTS` access below happens strictly between a matching
/// `pool_lock()`/`pool_unlock()` pair — see `POOL_LOCK`'s own doc comment.
fn pool_lock() {
    use std::sync::atomic::Ordering::{Acquire, Relaxed};
    while POOL_LOCK.compare_exchange_weak(false, true, Acquire, Relaxed).is_err() {
        std::hint::spin_loop();
    }
}
fn pool_unlock() {
    POOL_LOCK.store(false, std::sync::atomic::Ordering::Release);
}

/// Bytes of freed blocks one thread keeps per size class before handing the
/// excess back to the shared lists (`FREE_LISTS`). In bytes, not blocks.
/// Small on purpose: with tasks (`doc/plan-spawn.md`), blocks are often
/// allocated on one thread and freed on another (a task's result, released by
/// its parent), so the freeing thread's cache fills while the allocating
/// threads find theirs empty — a cache sized for one thread alone hoarded
/// hundreds of megabytes that way.
const THREAD_CACHE_BYTES: usize = 512 << 10;

/// Blocks above this size skip the thread caches entirely: the depot's lock
/// costs nothing next to what a block this large is used for, and it's these
/// that are expensive to hoard (tcmalloc's large-object rule).
const THREAD_CACHE_MAX_BLOCK: usize = 256 << 10;

/// `CLEAVE_NO_THREAD_CACHE=1`: every block goes straight to the depot, so the
/// bytes still allocated are exactly the live ones — what a leak test needs
/// (`cleave/tests/spawn_leaks.rs`).
static NO_THREAD_CACHE: std::sync::LazyLock<bool> =
    std::sync::LazyLock::new(|| std::env::var_os("CLEAVE_NO_THREAD_CACHE").is_some());

/// Blocks of `class` a thread keeps: `THREAD_CACHE_BYTES` per doubling of
/// size, shared by its `1 << CLASS_STEP_BITS` classes. Per class, the finer
/// classes multiplied what a thread could hoard by as many.
fn thread_cache_cap(class: usize) -> usize {
    ((THREAD_CACHE_BYTES >> CLASS_STEP_BITS) / class_bytes(class)).max(1)
}

/// Whether blocks of `class` go through the thread caches at all.
fn thread_cached(class: usize) -> bool {
    class_bytes(class) <= THREAD_CACHE_MAX_BLOCK && !*NO_THREAD_CACHE
}

/// One block of `class` from the depot, if any.
///
/// # Safety
///
/// `class < NUM_SIZE_CLASSES`.
unsafe fn depot_pop(class: usize) -> Option<*mut u8> {
    pool_lock();
    let b = unsafe { FREE_LISTS[class] };
    if !b.is_null() {
        unsafe { FREE_LISTS[class] = *(b as *mut *mut u8) };
    }
    pool_unlock();
    (!b.is_null()).then_some(b)
}

/// `block` back to the depot.
///
/// # Safety
///
/// `block` is a freed block of `class`, owned by no one.
unsafe fn depot_push(class: usize, block: *mut u8) {
    pool_lock();
    unsafe {
        *(block as *mut *mut u8) = FREE_LISTS[class];
        FREE_LISTS[class] = block;
    }
    pool_unlock();
}

/// Each thread's own free lists, in front of the shared `FREE_LISTS`
/// (which become the depot): an allocation pops from its thread's list
/// without any lock, a release pushes onto the releasing thread's list. The
/// shared lock is taken once per batch — refilling an empty list, or giving
/// back half of one past `thread_cache_cap` — instead of once per call,
/// which on many threads (`doc/plan-spawn.md`, tasks) would serialize every
/// allocation on one spinlock. Blocks migrate freely between threads through
/// the depot; the cap keeps a thread that frees more than it allocates (a
/// parent receiving its tasks' results) from hoarding. A thread's lists go
/// back to the depot when it exits.
struct ThreadCache {
    head: [std::cell::Cell<*mut u8>; NUM_SIZE_CLASSES],
    count: [std::cell::Cell<usize>; NUM_SIZE_CLASSES],
}

impl Drop for ThreadCache {
    fn drop(&mut self) {
        for class in 0..NUM_SIZE_CLASSES {
            let n = self.count[class].get();
            if n > 0 {
                unsafe { move_to_depot(&self.head[class], &self.count[class], class, n) };
            }
        }
    }
}

thread_local! {
    static THREAD_CACHE: ThreadCache = const {
        ThreadCache {
            head: [const { std::cell::Cell::new(std::ptr::null_mut()) }; NUM_SIZE_CLASSES],
            count: [const { std::cell::Cell::new(0) }; NUM_SIZE_CLASSES],
        }
    };
}

/// Moves `n` blocks from the front of a thread list to the depot, under one
/// lock.
///
/// # Safety
///
/// The list holds at least `n` blocks, each at least a pointer in size.
unsafe fn move_to_depot(head: &std::cell::Cell<*mut u8>, count: &std::cell::Cell<usize>, class: usize, n: usize) {
    pool_lock();
    for _ in 0..n {
        let block = head.get();
        unsafe {
            head.set(*(block as *mut *mut u8));
            *(block as *mut *mut u8) = FREE_LISTS[class];
            FREE_LISTS[class] = block;
        }
    }
    pool_unlock();
    count.set(count.get() - n);
}

/// A cached block of `class`, if any: this thread's list first, then a batch
/// from the depot.
fn pool_pop(class: usize) -> Option<*mut u8> {
    if !thread_cached(class) {
        return unsafe { depot_pop(class) };
    }
    let local = THREAD_CACHE.try_with(|c| unsafe {
        let head = &c.head[class];
        let block = head.get();
        if !block.is_null() {
            head.set(*(block as *mut *mut u8));
            c.count[class].set(c.count[class].get() - 1);
            return Some(block);
        }
        // Refill: one block to return, up to half a cache's worth kept.
        let want = (thread_cache_cap(class) / 2).max(1);
        let mut first = None;
        pool_lock();
        for _ in 0..want {
            let b = FREE_LISTS[class];
            if b.is_null() {
                break;
            }
            FREE_LISTS[class] = *(b as *mut *mut u8);
            if first.is_none() {
                first = Some(b);
            } else {
                *(b as *mut *mut u8) = head.get();
                head.set(b);
                c.count[class].set(c.count[class].get() + 1);
            }
        }
        pool_unlock();
        first
    });
    match local {
        Ok(found) => found,
        // This thread's cache is already gone (thread exit): the depot alone.
        Err(_) => unsafe { depot_pop(class) },
    }
}

/// Caches the freed `block` of `class`.
///
/// # Safety
///
/// `block` is a freed block of `class`, owned by no one.
unsafe fn pool_push(class: usize, block: *mut u8) {
    if !thread_cached(class) {
        return unsafe { depot_push(class, block) };
    }
    let pushed = THREAD_CACHE.try_with(|c| unsafe {
        *(block as *mut *mut u8) = c.head[class].get();
        c.head[class].set(block);
        let n = c.count[class].get() + 1;
        c.count[class].set(n);
        let cap = thread_cache_cap(class);
        if n > cap {
            move_to_depot(&c.head[class], &c.count[class], class, n - cap / 2);
        }
    });
    if pushed.is_err() {
        unsafe { depot_push(class, block) };
    }
}

/// Size classes per doubling, as `2^CLASS_STEP_BITS`: a block is at most
/// `1 / 2^CLASS_STEP_BITS` larger than the request (12.5%), where powers of
/// two lost up to half of it. Measured on nanoLM (`CLEAVE_ALLOC_STATS`): the
/// tensors of a model come in a few shapes, and its powers of two plus the
/// 64-byte header (a 4 MiB tensor, a 16 MiB one) each took a block of twice
/// their size; the live bytes at a step's peak, 5.8 GiB, held 9.6 GiB.
const CLASS_STEP_BITS: u32 = 3;

/// Every class a 64-bit size can fall in (`size_class`): 64 doublings of
/// `2^CLASS_STEP_BITS` classes. The unused ones cost a pointer-sized slot
/// each.
const NUM_SIZE_CLASSES: usize = 64 << CLASS_STEP_BITS;

/// One intrusive singly-linked free list per size class — `FREE_LISTS[c]`
/// is the most-recently-released block's own base pointer (the header's own
/// address, `cleave_release`'s own `rc_header`), or null if none is
/// currently cached. The "next" pointer for each link is stored *in* the
/// freed block itself, overwriting the now-dead `RcHeader` (every class's
/// physical allocation is at least 32 bytes, `size_class`'s own `.max(32)`
/// floor — always room for one `*mut u8`) — no separate free-list node type
/// or allocation needed, the classic segregated-free-list trick.
static mut FREE_LISTS: [*mut u8; NUM_SIZE_CLASSES] = [std::ptr::null_mut(); NUM_SIZE_CLASSES];

/// The class of a block of `total.max(32)` bytes: the smallest class whose
/// blocks (`class_bytes`) hold it. A class is a doubling (the exponent of
/// the highest bit of `total - 1`) and a step within it (the
/// `CLASS_STEP_BITS` bits below that one): a block is `2^e` plus `step + 1`
/// steps of `2^e / 2^CLASS_STEP_BITS`. The `32` floor matches
/// `RC_HEADER_SIZE` (16 bytes) plus a little real payload room being the
/// smallest allocation this runtime ever actually makes, and guarantees
/// every cached block has at least 8 bytes free for its own free-list
/// "next" pointer even at `data_size == 0`.
fn size_class(total: usize) -> usize {
    let x = total.max(32) - 1;
    let e = usize::BITS - 1 - x.leading_zeros();
    let step = (x >> (e - CLASS_STEP_BITS)) & ((1 << CLASS_STEP_BITS) - 1);
    ((e as usize) << CLASS_STEP_BITS) | step
}

/// The physical size actually allocated (and, symmetrically, freed) for
/// every block in size class `c` — a pure function of `c` alone, so a block
/// popped from `FREE_LISTS[c]` at alloc time and one pushed back at release
/// time always agree on layout, even though the *requested* `data_size`
/// generally differs from one occupant of the class to the next (the whole
/// point of bucketing instead of tracking exact sizes: a slightly smaller
/// same-class request still reuses the block cleanly).
fn class_bytes(class: usize) -> usize {
    let e = (class >> CLASS_STEP_BITS) as u32;
    let step = class & ((1 << CLASS_STEP_BITS) - 1);
    ((1 << CLASS_STEP_BITS) + step + 1) << (e - CLASS_STEP_BITS)
}

/// Data alignment for any allocation of at least this many bytes: one
/// AVX-512 vector, one cache line. A tensor payload aligned to only 16 bytes
/// (what every allocator here used to give) makes every 64-byte vector load
/// straddle two cache lines -- measured with AMD uProf on `examples/mnist-
/// interop`: ~1 load in 3 was cache-line-crossing (`MISALIGNED_LOADS`
/// ~224 per 1000 instructions), the weights being the worst case since
/// they are the most-read data and live in `cleave_alloc_rc` field buffers.
const VECTOR_ALIGN: usize = 64;

/// Offset of a headered block's own data from the block's base. The
/// `RcHeader` always sits immediately before the data (`rc_header`'s
/// `ptr - RC_HEADER_SIZE` never changes); for a payload of at least
/// `VECTOR_ALIGN` bytes, 48 bytes of padding go in front of it so the data
/// itself lands on a 64-byte boundary. A smaller payload gains nothing from
/// that (it fits in one line at 16) and keeps the compact layout. A pure
/// function of `data_size`, which the header records, so a release can
/// always find the block's base again.
fn data_offset(data_size: usize) -> usize {
    if data_size >= VECTOR_ALIGN {
        VECTOR_ALIGN
    } else {
        RC_HEADER_SIZE
    }
}

/// Alignment of every block in size class `class`. Uniform per class, not
/// per request: `FREE_LISTS` hands a class's blocks to headered and
/// headerless allocations alike, so every block of a class must already
/// satisfy the strictest use either could make of it.
fn class_align(class: usize) -> usize {
    if class_bytes(class) >= VECTOR_ALIGN {
        VECTOR_ALIGN
    } else {
        16
    }
}

/// Base of the block `header` belongs to -- the address `FREE_LISTS`, the
/// real `dealloc`, and every debug-bookkeeping table (`PARKED`, alloc
/// serials) identify a block by.
///
/// # Safety
/// `header` must be a live (or parked, whose `data_size` stays intact)
/// `RcHeader` this runtime wrote.
unsafe fn block_base(header: *mut RcHeader) -> *mut u8 {
    unsafe {
        let data = (header as *mut u8).add(RC_HEADER_SIZE);
        data.sub(data_offset((*header).data_size as usize))
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn cleave_alloc_rc(data_size: i64) -> *mut u8 {
    let offset = data_offset(data_size as usize);
    let total = offset + data_size as usize;
    let class = size_class(total);
    unsafe {
        // Pop under `POOL_LOCK` (`POOL_LOCK`'s own doc comment: real,
        // concurrent OpenMP-worker traffic reaches this exact spot) --
        // released again before the fallback `std::alloc::alloc` below,
        // which never touches `FREE_LISTS` at all and has no reason to
        // serialize against it.
        let popped = if class < NUM_SIZE_CLASSES {
            let popped = pool_pop(class);
            // Debug-only `PARKED` bookkeeping: the popped block now belongs
            // to this thread alone, so no other thread can push or pop it
            // while this runs (`pool_push` marks a block parked *before*
            // caching it, for the same reason).
            if *CLEAVE_DEBUG_POOL {
                if let Some(block) = popped {
                    if !parked_remove(block as usize) && *CLEAVE_DEBUG_POOL {
                        eprintln!(
                            "CLEAVE_DEBUG_POOL: popped block {block:p} was not marked parked (pool corruption)"
                        );
                    }
                }
            }
            popped
        } else {
            None
        };
        let base = match popped {
            Some(block) => block,
            None => {
                let layout = std::alloc::Layout::from_size_align(class_bytes(class), class_align(class))
                    .expect("cleave_alloc_rc: invalid layout");
                let p = std::alloc::alloc(layout);
                // The request's size tells an exhausted system (a plausible
                // size) from a corrupted request (an absurd one).
                assert!(
                    !p.is_null(),
                    "cleave_alloc_rc: allocation failed: {data_size} bytes requested, size class {class} ({} bytes)",
                    class_bytes(class)
                );
                p
            }
        };
        if *alloc_stats::ENABLED {
            alloc_stats::record(data_size as usize, popped.is_none(), Some((base as usize, class_bytes(class))));
        }
        let header = base.add(offset - RC_HEADER_SIZE) as *mut RcHeader;
        (*header).refcount = 1;
        (*header).data_size = data_size;
        if *CLEAVE_TRACE_RC {
            let serial = record_alloc_serial(base as usize);
            eprintln!("ALLOC   {base:p} #{serial}  {}", first_cleave_frame());
        }
        if *CLEAVE_TRACE_SIZE == Some(data_size) {
            eprintln!(
                "TRACE_SIZE alloc_rc  {base:p} size={data_size}  {}",
                cleave_frames()
            );
        }
        base.add(offset)
    }
}

/// # Safety
/// See `rc_header`'s own safety contract — `ptr` must be a live `cleave_
/// alloc_rc` result.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cleave_retain(ptr: *mut u8) {
    unsafe {
        let header = rc_header(ptr);
        let base = block_base(header);
        if *CLEAVE_DEBUG_POOL && parked_contains(base as usize) {
            eprintln!("CLEAVE_DEBUG_POOL: cleave_retain on parked (already-freed) block {:p}", base);
        }
        if *CLEAVE_TRACE_RC {
            eprintln!(
                "RETAIN  {:p} #{}  -> {}  {}",
                base,
                current_alloc_serial(base as usize).map_or("?".to_string(), |s| s.to_string()),
                (*header).refcount + 1,
                first_cleave_frame()
            );
        }
        if *CLEAVE_TRACE_SIZE == Some((*header).data_size) {
            eprintln!(
                "TRACE_SIZE retain    {:p} size={} rc {} -> {}  {}",
                header,
                (*header).data_size,
                (*header).refcount,
                (*header).refcount + 1,
                first_cleave_frame()
            );
        }
        refcount_of(header).fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

/// `header`'s refcount as an atomic: atomic because `spawn`'s tasks
/// (`doc/plan-spawn.md`) share objects across threads. The field stays a
/// plain `i64` in `RcHeader` (initialized non-atomically at allocation, before
/// the block is visible to anyone else); every later access goes through here.
/// The orderings are `Arc`'s: an increment needs none (whoever increments
/// already holds a reference), a decrement publishes this thread's writes
/// (`Release`), and the thread that frees takes them all (`Acquire`).
///
/// # Safety
///
/// `header` must be a live block's header.
unsafe fn refcount_of<'a>(header: *mut RcHeader) -> &'a std::sync::atomic::AtomicI64 {
    unsafe { std::sync::atomic::AtomicI64::from_ptr(std::ptr::addr_of_mut!((*header).refcount)) }
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
///
/// **A real, reproducible instance of exactly this hazard is still open**
/// (`doc/backlog.md`'s own double-release entry) — tried, and reverted,
/// making the `CLEAVE_DEBUG_POOL`-only detection below into an always-on
/// production safety net (silently no-op the redundant release instead of
/// crashing): a global `Mutex<HashSet>` touched on *every* alloc/release,
/// not just the buggy ones, measured a real ~50-100x slowdown on
/// `mnist-interop`, and the underlying bug turned out to also leak memory
/// for real (a genuine `cleave_alloc_rc: allocation failed` a few epochs
/// in, once the crash itself stopped masking it) — never a full fix to
/// begin with, and not worth that cost for a partial one. Left `CLEAVE_
/// DEBUG_POOL`-gated, as originally built, until the real extra release
/// call is found and removed at the source.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cleave_release(ptr: *mut u8) -> bool {
    unsafe {
        let header = rc_header(ptr);
        let base = block_base(header);
        if *DIAGNOSTICS {
            if *CLEAVE_DEBUG_POOL && parked_contains(base as usize) {
                eprintln!(
                    "CLEAVE_DEBUG_POOL: cleave_release on parked (already-freed) block {:p} #{}, refcount={}, data_size={}",
                    base,
                    current_alloc_serial(base as usize).map_or("?".to_string(), |s| s.to_string()),
                    (*header).refcount,
                    // `data_size` survives being parked untouched -- the free-
                    // list's own "next" link overwrites only the block's first
                    // 8 bytes (at most `refcount`'s own slot, for a compact
                    // block), so this is still the block's real original
                    // allocation size, a real clue to which tensor shape this is.
                    (*header).data_size,
                );
                // The *offending* release's own entry point and source position
                // -- the one thing this message was missing to be directly
                // actionable. `CLEAVE_TRACE_SIZE`'s own ledger already gives
                // every *earlier* allocate/retain/release site for the same
                // block, so this closes the loop: which release is the second
                // one, and which of the two ownership systems emitted it.
                eprintln!(
                    "CLEAVE_DEBUG_POOL:   offending release via {} at {}",
                    release_entry_point(),
                    first_cleave_frame()
                );
                // A real breakpoint exception, not `backtrace`'s own runtime
                // walk -- that crate hits a real, unavoidable limit for a call
                // site inside a function whose own Win64 unwind info isn't
                // registered at this FFI boundary (deeply-inlined/vectorized
                // functions, mostly). Now that real debug info is generated
                // (`pipeline.rs`'s own `DISubprogram` emission), a debugger
                // attached at this exact point (`cdb -g -G -c "g;kb;q"`, say)
                // resolves the *real* call stack, inlined frames included, off
                // the PDB directly -- no such limitation. Falls through to the
                // same `exit(97)` when nothing is attached to catch it.
                #[cfg(target_arch = "x86_64")]
                std::arch::asm!("int3");
                std::process::exit(97);
            }
            if *CLEAVE_TRACE_RC {
                eprintln!(
                    "RELEASE {:p} #{}  -> {}  {}",
                    base,
                    current_alloc_serial(base as usize).map_or("?".to_string(), |s| s.to_string()),
                    (*header).refcount - 1,
                    first_cleave_frame()
                );
            }
            if *CLEAVE_TRACE_SIZE == Some((*header).data_size) {
                eprintln!(
                    "TRACE_SIZE release[{}] {:p} size={} rc {} -> {} in_arena={}  {}",
                    release_entry_point(),
                    header,
                    (*header).data_size,
                    (*header).refcount,
                    (*header).refcount - 1,
                    is_in_arena(header as *mut u8),
                    cleave_frames()
                );
            }
        }
        if refcount_of(header).fetch_sub(1, std::sync::atomic::Ordering::Release) == 1 {
            std::sync::atomic::fence(std::sync::atomic::Ordering::Acquire);
            // Arena-backed (`cleave_alloc_rc`'s own doc comment): never
            // individually freed here — the matching `cleave_region_exit`
            // reclaims it in bulk, along with everything else allocated
            // since. Still correctly reports `true` below either way: a
            // struct's own cascading release into its refcounted fields
            // (`mlir_lower.rs::lower_release_cascade`) must still fire
            // once *this* container's own count genuinely reaches zero,
            // regardless of which physical allocator backs it.
            if !is_in_arena(header as *mut u8) {
                let data_size = (*header).data_size as usize;
                let total = data_offset(data_size) + data_size;
                let class = size_class(total);
                if class < NUM_SIZE_CLASSES {
                    // Cache it instead of returning it to the OS heap --
                    // `cleave_alloc_rc`'s own `FREE_LISTS` doc comment.
                    // `class_bytes(class)` is what actually got allocated
                    // for this block (`cleave_alloc_rc`'s own symmetric
                    // rounding), so writing the free-list "next" pointer
                    // into its first 8 bytes is always in-bounds.

                    if *CLEAVE_DEBUG_POOL && !parked_insert(base as usize) {
                        eprintln!("CLEAVE_DEBUG_POOL: block {base:p} parked twice (double-free)");
                    }
                    if *alloc_stats::ENABLED {
                        alloc_stats::forget(base as usize, true);
                    }
                    pool_push(class, base);
                } else {
                    if *alloc_stats::ENABLED {
                        alloc_stats::forget(base as usize, false);
                    }
                    // Astronomically large (`class >= 64`, i.e. `total >
                    // 2^63`) -- can't happen with a real `i64 data_size`,
                    // but falls back to the plain, uncached path rather
                    // than indexing out of bounds if it ever somehow did.
                    let layout = std::alloc::Layout::from_size_align(class_bytes(class), class_align(class))
                        .expect("cleave_release: invalid layout");
                    std::alloc::dealloc(base, layout);
                }
            }
            true
        } else {
            false
        }
    }
}

/// Bytes of one thread's arena (`doc/hld.md`'s own "Memory management"
/// section: "one large reserved VM region... pages committed lazily").
/// Reserved as address space once for every thread (`ARENA_SLOTS`, below),
/// each thread's slice committed as its cursor first reaches it, by
/// `ARENA_COMMIT_STEP`. 256 MiB: bigger
/// than any single training-loop iteration's own local footprint this
/// project's own real workload (`examples/mnist-interop`, per-sample
/// tensors well under a megabyte) plausibly needs — a real number to
/// revisit once a real workload's own peak region depth is measured, not
/// a permanent ceiling; `cleave_alloc_local`'s own overflow check exists
/// specifically so exceeding it fails loudly rather than corrupting
/// whatever memory happens to sit past the reserved region.
const ARENA_CAPACITY: usize = 256 * 1024 * 1024;

/// The arenas: one per thread, each a `ARENA_CAPACITY` slice of one address
/// range reserved once for all of them (`ARENA_SLOTS` slices, reserved, not
/// committed: no memory is used until a thread opens its first region and
/// commits its own slice). Per thread because a region is a stack discipline
/// (`cleave_region_exit` rewinds the cursor to its `cleave_region_enter`
/// handle): with one global arena, a `spawn` task on another thread
/// (`doc/plan-spawn.md`) would rewind over this thread's live allocations.
/// One reserved range, rather than one allocation per thread, keeps
/// `is_in_arena` a single bounds check, whichever thread allocated the block
/// and whichever releases it. A thread hands its slice back when it exits
/// (the test harnesses start a thread per test), committed memory kept for
/// the next one.
const ARENA_SLOTS: usize = 256;

/// An arena slice is committed by this many bytes at a time, as its cursor
/// first reaches them. Committed whole when a thread first opened a region,
/// nanoLM's threads held 2 GiB of committed memory, almost none of it ever
/// touched: most open a region and never allocate in it; the one that does
/// uses under 1 MiB (`CLEAVE_ALLOC_STATS`: peak private bytes 10.1 -> 8.0
/// GiB).
const ARENA_COMMIT_STEP: usize = 1024 * 1024;

/// Bytes committed in each slice, by slot: a slice handed back by an exiting
/// thread keeps its pages for the next one.
static ARENA_COMMITTED: std::sync::Mutex<Vec<usize>> = std::sync::Mutex::new(Vec::new());

/// The slices taken so far and the bytes committed in them.
fn arena_commitment() -> (usize, usize) {
    let committed = ARENA_COMMITTED.lock().unwrap_or_else(|e| e.into_inner());
    (committed.iter().filter(|&&c| c > 0).count(), committed.iter().sum())
}

static ARENA_RESERVATION: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
static FREE_ARENA_SLOTS: std::sync::Mutex<Vec<usize>> = std::sync::Mutex::new(Vec::new());
static NEXT_ARENA_SLOT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// This thread's arena: its slice's base (0 until its first region), the bump
/// cursor (an offset into the slice) and the number of open regions.
struct ThreadArena {
    base: std::cell::Cell<usize>,
    cursor: std::cell::Cell<usize>,
    depth: std::cell::Cell<usize>,
    /// Bytes of the slice committed (`ARENA_COMMIT_STEP` by step).
    committed: std::cell::Cell<usize>,
}

impl Drop for ThreadArena {
    fn drop(&mut self) {
        let base = self.base.get();
        if base != 0 {
            let slot = (base - arena_reservation()) / ARENA_CAPACITY;
            FREE_ARENA_SLOTS.lock().unwrap_or_else(|e| e.into_inner()).push(slot);
        }
    }
}

thread_local! {
    static ARENA: ThreadArena = const {
        ThreadArena {
            base: std::cell::Cell::new(0),
            cursor: std::cell::Cell::new(0),
            depth: std::cell::Cell::new(0),
            committed: std::cell::Cell::new(0),
        }
    };
}

#[cfg(windows)]
mod arena_memory {
    const MEM_COMMIT: u32 = 0x1000;
    const MEM_RESERVE: u32 = 0x2000;
    const PAGE_NOACCESS: u32 = 0x01;
    const PAGE_READWRITE: u32 = 0x04;
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn VirtualAlloc(address: *mut std::ffi::c_void, size: usize, kind: u32, protect: u32) -> *mut std::ffi::c_void;
    }
    pub fn reserve(size: usize) -> usize {
        let p = unsafe { VirtualAlloc(std::ptr::null_mut(), size, MEM_RESERVE, PAGE_NOACCESS) };
        assert!(!p.is_null(), "cleave arena: reserving {size} bytes of address space failed");
        p as usize
    }
    pub fn commit(base: usize, size: usize) {
        let p = unsafe { VirtualAlloc(base as *mut _, size, MEM_COMMIT, PAGE_READWRITE) };
        assert!(!p.is_null(), "cleave arena: committing {size} bytes failed");
    }
}

#[cfg(unix)]
mod arena_memory {
    const PROT_READ: i32 = 1;
    const PROT_WRITE: i32 = 2;
    const MAP_PRIVATE: i32 = 0x02;
    #[cfg(target_os = "macos")]
    const MAP_ANONYMOUS: i32 = 0x1000;
    #[cfg(not(target_os = "macos"))]
    const MAP_ANONYMOUS: i32 = 0x20;
    #[cfg(target_os = "linux")]
    const MAP_NORESERVE: i32 = 0x4000;
    #[cfg(not(target_os = "linux"))]
    const MAP_NORESERVE: i32 = 0;
    unsafe extern "C" {
        fn mmap(addr: *mut std::ffi::c_void, len: usize, prot: i32, flags: i32, fd: i32, off: i64) -> *mut std::ffi::c_void;
    }
    /// Readable and writable from the start: pages are only backed when
    /// touched, so committing is a no-op.
    pub fn reserve(size: usize) -> usize {
        let p = unsafe {
            mmap(std::ptr::null_mut(), size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0)
        };
        assert!(p as isize != -1, "cleave arena: reserving {size} bytes of address space failed");
        p as usize
    }
    pub fn commit(_base: usize, _size: usize) {}
}

fn arena_reservation() -> usize {
    *ARENA_RESERVATION.get_or_init(|| arena_memory::reserve(ARENA_SLOTS * ARENA_CAPACITY))
}

/// This thread's arena slice, taking and committing one on first use.
fn arena_base() -> *mut u8 {
    ARENA.with(|a| {
        if a.base.get() == 0 {
            let reused = FREE_ARENA_SLOTS.lock().unwrap_or_else(|e| e.into_inner()).pop();
            let slot = match reused {
                Some(slot) => slot,
                None => {
                    let slot = NEXT_ARENA_SLOT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    assert!(slot < ARENA_SLOTS, "cleave arena: more than {ARENA_SLOTS} threads with an open region");
                    slot
                }
            };
            let committed = {
                let mut all = ARENA_COMMITTED.lock().unwrap_or_else(|e| e.into_inner());
                if all.len() <= slot {
                    all.resize(slot + 1, 0);
                }
                all[slot]
            };
            a.base.set(arena_reservation() + slot * ARENA_CAPACITY);
            a.cursor.set(0);
            a.committed.set(committed);
        }
        a.base.get() as *mut u8
    })
}

fn arena_bump(size: usize, align: usize) -> *mut u8 {
    let base = arena_base();
    ARENA.with(|a| {
        let cursor = a.cursor.get();
        let aligned = (cursor + align - 1) & !(align - 1);
        let new_cursor = aligned + size;
        assert!(
            new_cursor <= ARENA_CAPACITY,
            "cleave arena exhausted ({new_cursor} > {ARENA_CAPACITY} bytes) -- \
             a real overflow path (grow, or fall back to the ordinary allocator) is not built yet"
        );
        if new_cursor > a.committed.get() {
            let committed = a.committed.get();
            let target = new_cursor.div_ceil(ARENA_COMMIT_STEP) * ARENA_COMMIT_STEP;
            arena_memory::commit(base as usize + committed, target - committed);
            a.committed.set(target);
            let slot = (base as usize - arena_reservation()) / ARENA_CAPACITY;
            ARENA_COMMITTED.lock().unwrap_or_else(|e| e.into_inner())[slot] = target;
        }
        a.cursor.set(new_cursor);
        unsafe { base.add(aligned) }
    })
}

/// Whether `ptr` lies in any thread's arena: one bounds check on the shared
/// reservation.
fn is_in_arena(ptr: *mut u8) -> bool {
    let Some(&base) = ARENA_RESERVATION.get() else { return false };
    let addr = ptr as usize;
    addr >= base && addr < base + ARENA_SLOTS * ARENA_CAPACITY
}

#[unsafe(no_mangle)]
pub extern "C" fn cleave_region_enter(_size: i64) -> i64 {
    arena_base();
    ARENA.with(|a| {
        a.depth.set(a.depth.get() + 1);
        a.cursor.get() as i64
    })
}

/// `doc/hld.md`'s own `alloc_local(handle, size) -> ptr` — carves `size`
/// bytes out of the arena at the current cursor (laid out exactly like a
/// `cleave_alloc_rc` block: `data_offset`, with the data 64-byte aligned for
/// a payload of at least `VECTOR_ALIGN` bytes), bumps the cursor forward. `handle` (the region this
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
    assert!(
        ARENA.with(|a| a.depth.get()) > 0,
        "cleave_alloc_local called with no region open -- a real compiler bug, \
         never a legitimate runtime condition"
    );
}
#[unsafe(no_mangle)]
pub extern "C" fn cleave_alloc_local(_handle: i64, size: i64) -> *mut u8 {
    assert_region_open();
    if *alloc_stats::ENABLED {
        alloc_stats::record(size as usize, false, None);
    }
    let offset = data_offset(size as usize);
    let align = if offset == VECTOR_ALIGN { VECTOR_ALIGN } else { 16 };
    let base = arena_bump(offset + size as usize, align);
    unsafe {
        let header = base.add(offset - RC_HEADER_SIZE) as *mut RcHeader;
        (*header).refcount = 1;
        (*header).data_size = size;
        if *CLEAVE_TRACE_SIZE == Some(size) {
            eprintln!(
                "TRACE_SIZE alloc_local {base:p} size={size}  {}",
                first_cleave_frame()
            );
        }
        base.add(offset)
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
    ARENA.with(|a| {
        a.cursor.set(handle as usize);
        a.depth.set(a.depth.get() - 1);
    });
}

/// `cleave_release`'s own `bool` result ("did this call actually free the
/// block"), discarded — matches `free`'s own `(ptr) -> ()` C signature
/// exactly. Exists purely so `unify_alloc.rs`'s own `llvm.call @free` ->
/// `llvm.call @cleave_release_void` rewrite can be a **plain callee-symbol
/// rename**, nothing else: melior's own `remove_from_parent` is confirmed
/// unsafe to call at all on real ops from this pipeline (found first on
/// `memcpy`, and — checked again here, since a *
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
    RELEASE_VIA_VOID.with(|f| f.set(true));
    unsafe {
        cleave_release(ptr);
    }
    RELEASE_VIA_VOID.with(|f| f.set(false));
}

/// The other half of `doc/plan-affine-ownership.md`'s Stage 2, alongside
/// `cleave_release_pool` below: a heap allocation with **no `RcHeader` at
/// all** — no refcount, no retain/release machinery, just `data_size`
/// bytes of storage. Only ever emitted by `mlir_lower.rs` for a
/// `PrimOp::Struct` construction the compiler has *proven* — statically,
/// via `alias_analysis::value_is_ever_aliased` — can never have more than
/// one live reference, and whose own field shape has no refcounted/tensor
/// field of its own to cascade into (the *first* landing of this
/// mechanism is deliberately restricted to that simpler case; a struct
/// embedding another refcounted value needs a real cascade story worked
/// out before it can go through here too — not yet built).
///
/// **Shares `FREE_LISTS`/`size_class`/`class_bytes` with `cleave_alloc_rc`
/// unconditionally, the exact same buckets** — deliberate, not incidental:
/// a block a headerless release just returned to a size class is exactly
/// as reusable by a *headered* allocation of the same class as the
/// reverse, since each allocator writes everything it needs (a header, or
/// nothing) fresh at allocation time and never assumes anything about a
/// popped block's own previous occupant. Splitting into two disjoint
/// pools would only fragment the cache for no safety benefit.
///
/// No header means no `data_size` stored anywhere for `cleave_release_
/// pool` to read back later — **the caller must pass it again** at
/// release time (`cleave_release_pool`'s own doc comment). This is not
/// extra bookkeeping the runtime is missing: for a value this analysis
/// has already proven affine, both its size *and* its release point are
/// static, compile-time facts (`mlir_lower.rs` already computes the exact
/// same `sizeof` at the allocation site via `llvm_type_size_bytes`) — the
/// entire point of proving "never aliased" is that nothing about this
/// value's own lifetime needs to be tracked at runtime at all.
#[unsafe(no_mangle)]
pub extern "C" fn cleave_alloc_pool(data_size: i64) -> *mut u8 {
    let total = data_size.max(0) as usize;
    let class = size_class(total);
    unsafe {
        let popped = if class < NUM_SIZE_CLASSES {
            pool_pop(class)
        } else {
            None
        };
        if *alloc_stats::ENABLED {
            alloc_stats::record(total, popped.is_none(), None);
        }
        match popped {
            Some(block) => block,
            None => {
                let layout = std::alloc::Layout::from_size_align(class_bytes(class), class_align(class))
                    .expect("cleave_alloc_pool: invalid layout");
                let p = std::alloc::alloc(layout);
                assert!(!p.is_null(), "cleave_alloc_pool: allocation failed");
                p
            }
        }
    }
}

/// The release half of `cleave_alloc_pool` above — see that function's own
/// doc comment for why `data_size` must be passed again here (no header
/// stores it), and why sharing `FREE_LISTS` with `cleave_alloc_rc` is
/// deliberate. Unconditional: no refcount to check, no cascade into
/// nested fields (`cleave_alloc_pool`'s own doc comment — restricted, for
/// now, to structs with no refcounted field of their own), just push the
/// block back to its size class, exactly once, matching the compiler's
/// own static proof that this is the value's one true last use.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cleave_release_pool(ptr: *mut u8, data_size: i64) {
    let total = data_size.max(0) as usize;
    let class = size_class(total);
    unsafe {
        if class < NUM_SIZE_CLASSES {
            pool_push(class, ptr);
        } else {
            let layout = std::alloc::Layout::from_size_align(class_bytes(class), class_align(class))
                .expect("cleave_release_pool: invalid layout");
            std::alloc::dealloc(ptr, layout);
        }
    }
}

// Which of the two entry points the release currently being serviced came
// in through -- and therefore **which of the two ownership systems emitted
// it**, which is the single most useful fact when diagnosing a double
// release. `cleave_release` is only ever emitted by cleave's own CPS
// refcounting (`refcount.rs`); `cleave_release_void` is only ever the
// rename of bufferization's own `free` (`unify_alloc.rs`). A block that
// receives one of each is, by definition, a CPS-vs-bufferization double-
// ownership bug. Read by `CLEAVE_TRACE_SIZE`'s ledger and by `CLEAVE_
// DEBUG_POOL`'s fatal message; thread-local because both entry points are
// genuinely reachable from several OpenMP workers at once (`POOL_LOCK`'s
// own doc comment).
thread_local! {
    static RELEASE_VIA_VOID: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

fn release_entry_point() -> &'static str {
    if RELEASE_VIA_VOID.with(|f| f.get()) {
        "void/bufferization"
    } else {
        "rc/CPS"
    }
}

// Always linked, unconditionally (`Cargo.toml`'s own doc comment on why).
// `stdlib/blas/blas.cleave`'s own `Sgemm::sgemm` is the one, explicit,
// low-level entry point to real BLAS (`doc/plan-blas-native.md` §7.1) --
// unlike the first attempt's own four wrappers (`cleave_blas_sgemm_
// rowmajor`/`transpose_a`/`transpose_b`/`bias_rowmajor`, each hardcoding
// its own `CBLAS_TRANSPOSE` flags), this is the *one* Rust-level symbol
// that exposes `cblas_sgemm`'s own real, generic shape directly --
// `trans_a`/`trans_b` are real runtime arguments here, not baked into
// which wrapper got called. Destination `c` is explicit, passed straight
// through -- no scratch buffer, nothing to redirect after the fact.
//
// **Explicit, lazy `LoadLibraryW`/`GetProcAddress` (`blas_dynload`,
// below), not an ordinary implicit `extern "C" { ... }` link against
// `openblas.lib` -- a real, load-bearing choice, not a stylistic one.**
// Windows resolves every *implicit* DLL import at process-*creation*
// time, before any of that process's own code ever runs -- found
// directly, the hard way: any build script that merely links `cleave-rt`
// transitively (`cleave-build`'s own `compile()`, called from every
// "-interop" example's own `build.rs`) would need `openblas.dll`
// discoverable the instant *its own* executable
// (`build-script-build.exe`, in an unpredictable, per-crate, hash-named
// `target/.../build/<pkg>-<hash>/` directory -- no single place to copy a
// DLL that covers every consumer) starts, *even though that process never
// actually calls a BLAS function at all* (object emission only ever
// builds and validates a JIT `ExecutionEngine`, never executes the
// generated code). Explicit, on-first-use loading sidesteps the whole
// problem: a process that never calls into this module never needs
// `openblas.dll` to exist at all, exactly the classic Windows answer to
// "an optional/location-variable runtime dependency shouldn't be a hard,
// process-startup import."
mod blas_dynload {
    use std::sync::OnceLock;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn LoadLibraryW(lp_lib_file_name: *const u16) -> *mut std::ffi::c_void;
        fn GetProcAddress(
            h_module: *mut std::ffi::c_void,
            lp_proc_name: *const u8,
        ) -> *mut std::ffi::c_void;
    }

    pub type CblasSgemmFn = unsafe extern "C" fn(
        order: i32,
        trans_a: i32,
        trans_b: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
    pub type OpenblasSetNumThreadsFn = unsafe extern "C" fn(num_threads: i32);

    pub struct BlasFns {
        pub cblas_sgemm: CblasSgemmFn,
        pub openblas_set_num_threads: OpenblasSetNumThreadsFn,
    }
    // Both fields are plain function pointers into a DLL that, once loaded,
    // stays mapped for the rest of the process's own lifetime -- sound to
    // share across threads the same way any other `'static fn` pointer is.
    unsafe impl Send for BlasFns {}
    unsafe impl Sync for BlasFns {}

    /// `OPENBLAS_PREFIX`, falling back to `<workspace root>/target/openblas`
    /// -- mirrors `cleave-rt/build.rs`'s own identical fallback exactly
    /// (`scripts/setup-openblas.ps1`'s own default `-CacheDir`).
    /// `env!("CARGO_MANIFEST_DIR")` is a *compile-time* macro -- the literal
    /// path is baked into this crate's own compiled code at the point
    /// `cleave-rt` itself was built, unaffected by wherever the *running*
    /// process (or its own current directory/`PATH`) happens to be later,
    /// which is exactly why this works regardless of which consuming
    /// process ends up loading it.
    fn openblas_dll_path() -> String {
        let prefix = std::env::var("OPENBLAS_PREFIX").unwrap_or_else(|_| {
            concat!(env!("CARGO_MANIFEST_DIR"), "/../target/openblas").to_string()
        });
        format!("{prefix}/bin/openblas.dll").replace('/', "\\")
    }

    fn load() -> BlasFns {
        let path = openblas_dll_path();
        let wide: Vec<u16> = path.encode_utf16().chain(std::iter::once(0)).collect();
        let handle = unsafe { LoadLibraryW(wide.as_ptr()) };
        if handle.is_null() {
            panic!(
                "cleave-rt: failed to load openblas.dll from {path} -- run \
                 scripts/setup-openblas.ps1, or set OPENBLAS_PREFIX"
            );
        }
        let sgemm = unsafe { GetProcAddress(handle, c"cblas_sgemm".as_ptr().cast()) };
        let set_threads =
            unsafe { GetProcAddress(handle, c"openblas_set_num_threads".as_ptr().cast()) };
        let (Some(sgemm), Some(set_threads)) = (
            std::ptr::NonNull::new(sgemm),
            std::ptr::NonNull::new(set_threads),
        ) else {
            panic!("cleave-rt: {path} loaded but is missing an expected symbol");
        };
        // SAFETY: both symbols were just resolved, by name, out of a real
        // OpenBLAS build (`cleave-openblas-redist`) whose own C ABI for
        // them is stable and already exercised directly (`bench_sgemm.c`).
        unsafe {
            BlasFns {
                cblas_sgemm: std::mem::transmute::<*mut std::ffi::c_void, CblasSgemmFn>(
                    sgemm.as_ptr(),
                ),
                openblas_set_num_threads: std::mem::transmute::<
                    *mut std::ffi::c_void,
                    OpenblasSetNumThreadsFn,
                >(set_threads.as_ptr()),
            }
        }
    }

    static FNS: OnceLock<BlasFns> = OnceLock::new();
    pub fn fns() -> &'static BlasFns {
        FNS.get_or_init(load)
    }
}

/// CBLAS's own `enum CBLAS_ORDER`/`enum CBLAS_TRANSPOSE` values (`cblas.h`)
/// -- hardcoded rather than bound via `bindgen`/a `-sys` crate: these are
/// the only values this thin wrapper ever needs, and they're a stable part
/// of the CBLAS C ABI, not something OpenBLAS's own build could change
/// between versions. `trans_a`/`trans_b` arrive here as real runtime `i32`s
/// straight from `stdlib/blas/blas.cleave`'s own `sgemm` (itself translating
/// a cleave-level `bool` into one of these two constants) -- this wrapper
/// itself makes no assumption about which one it'll get.
const CBLAS_ROW_MAJOR: i32 = 101;
const CBLAS_NO_TRANS: i32 = 111;
const CBLAS_TRANS: i32 = 112;

/// Pins OpenBLAS to a single thread, once, the first time `cleave_blas_
/// sgemm` is called -- a deliberate, temporary, hardcoded `1` (not yet a
/// real, tunable setting) to directly answer a concrete question: how does
/// a genuinely single-threaded OpenBLAS compare to cleave's own native
/// codegen, with the thread count actually pinned and verified rather than
/// left to whatever OpenBLAS's own default happens to be. `std::sync::
/// Once`, not a call on every GEMM invocation -- the API call itself is
/// presumably cheap, but there is no reason to pay it on every single call
/// when it only ever needs to run once per process. (`blas_dynload::fns()`'s
/// own `OnceLock` already guards the *load*; this is a separate, later step
/// -- the *call*.)
static PIN_BLAS_THREADS: std::sync::Once = std::sync::Once::new();
/// How many threads a parallel region opened for `spawn`'s tasks gets
/// (`doc/plan-spawn.md`; the region itself, `cleave-mlir-shim`'s
/// `cleaveLowerSpawns`): `OMP_NUM_THREADS` when set, otherwise one per
/// physical core. Not libomp's default of one per logical core: with SMT, the
/// sibling of a core running a task spins waiting for work and takes that
/// core's execution resources from it — nanoLM's data-parallel step, 8 threads
/// against 16 on an 8-core, 16-thread CPU, alternating runs in one session:
/// 275-276 against 290 ms/step, about 5% faster. Computed once.
#[unsafe(no_mangle)]
pub extern "C" fn cleave_parallel_threads() -> i32 {
    static THREADS: std::sync::OnceLock<i32> = std::sync::OnceLock::new();
    *THREADS.get_or_init(|| {
        let from_env = std::env::var("OMP_NUM_THREADS")
            .ok()
            .and_then(|v| v.split(',').next().and_then(|n| n.trim().parse::<i32>().ok()))
            .filter(|&n| n > 0);
        from_env.unwrap_or_else(|| {
            physical_cores()
                .or_else(|| std::thread::available_parallelism().ok().map(|n| n.get()))
                .unwrap_or(1) as i32
        })
    })
}

/// The machine's physical cores, each as the processor group and the mask of
/// its logical processors (its SMT siblings): one `RelationProcessorCore`
/// record each. Computed once.
#[cfg(windows)]
fn cores() -> Option<&'static [(u16, u64)]> {
    static CORES: std::sync::OnceLock<Option<Vec<(u16, u64)>>> = std::sync::OnceLock::new();
    CORES
        .get_or_init(|| {
            #[link(name = "kernel32")]
            unsafe extern "system" {
                fn GetLogicalProcessorInformationEx(relationship: u32, buffer: *mut u8, length: *mut u32) -> i32;
            }
            const RELATION_PROCESSOR_CORE: u32 = 0;
            let mut len = 0u32;
            unsafe { GetLogicalProcessorInformationEx(RELATION_PROCESSOR_CORE, std::ptr::null_mut(), &mut len) };
            if len == 0 {
                return None;
            }
            let mut buf = vec![0u8; len as usize];
            if unsafe { GetLogicalProcessorInformationEx(RELATION_PROCESSOR_CORE, buf.as_mut_ptr(), &mut len) } == 0 {
                return None;
            }
            // Each record: `Relationship: u32`, `Size: u32` (the record's own
            // size), then `PROCESSOR_RELATIONSHIP`: `Flags: u8`,
            // `EfficiencyClass: u8`, 20 reserved bytes, `GroupCount: u16`, then
            // `GroupCount` `GROUP_AFFINITY`s (`Mask: u64`, `Group: u16`, 6
            // reserved bytes); a core spans one group.
            let mut cores = Vec::new();
            let mut offset = 0usize;
            while offset + 8 <= len as usize {
                let size = u32::from_le_bytes(buf[offset + 4..offset + 8].try_into().ok()?) as usize;
                if size == 0 {
                    break;
                }
                let affinity = offset + 32;
                if affinity + 10 <= offset + size {
                    let mask = u64::from_le_bytes(buf[affinity..affinity + 8].try_into().ok()?);
                    let group = u16::from_le_bytes(buf[affinity + 8..affinity + 10].try_into().ok()?);
                    cores.push((group, mask));
                }
                offset += size;
            }
            (!cores.is_empty()).then_some(cores)
        })
        .as_deref()
}

#[cfg(windows)]
fn physical_cores() -> Option<usize> {
    cores().map(<[_]>::len)
}

#[cfg(not(windows))]
fn physical_cores() -> Option<usize> {
    None
}

/// Called by every member of a parallel team when the region starts (the
/// `omp.parallel` regions `cleave-mlir-shim`'s `cleaveBindTeams` marks), with
/// its number in the team: places the calling thread on physical core
/// `thread` (modulo their count), on all of that core's logical processors.
/// Left to the OS, two of a team's threads (one per physical core,
/// `cleave_parallel_threads`) can land on the two SMT siblings of one core;
/// every barrier then waits for that core, for the whole run: nanoLM's
/// training step drew ~1400 or ~1720 ms per process start, ~1330 for every
/// run once placed. Once per thread (a team's threads are kept from one
/// region to the next). Not when the placement is asked for explicitly
/// (`OMP_PLACES`, `OMP_PROC_BIND`, `KMP_AFFINITY`): libomp does it then.
#[unsafe(no_mangle)]
pub extern "C" fn cleave_bind_worker(thread: i32) {
    thread_local! {
        static BOUND: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }
    if BOUND.with(|b| b.replace(true)) {
        return;
    }
    static EXPLICIT: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        ["OMP_PLACES", "OMP_PROC_BIND", "KMP_AFFINITY"].iter().any(|v| std::env::var_os(v).is_some())
    });
    if *EXPLICIT {
        return;
    }
    bind_to_core(thread.max(0) as usize);
}

#[cfg(windows)]
fn bind_to_core(core: usize) {
    #[repr(C)]
    struct GroupAffinity {
        mask: u64,
        group: u16,
        reserved: [u16; 3],
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentThread() -> *mut std::ffi::c_void;
        fn SetThreadGroupAffinity(
            thread: *mut std::ffi::c_void,
            affinity: *const GroupAffinity,
            previous: *mut GroupAffinity,
        ) -> i32;
    }
    let Some(cores) = cores() else { return };
    let (group, mask) = cores[core % cores.len()];
    let affinity = GroupAffinity { mask, group, reserved: [0; 3] };
    unsafe { SetThreadGroupAffinity(GetCurrentThread(), &affinity, std::ptr::null_mut()) };
}

#[cfg(not(windows))]
fn bind_to_core(_core: usize) {}

fn ensure_thread_count_pinned() {
    PIN_BLAS_THREADS
        .call_once(|| unsafe { (blas_dynload::fns().openblas_set_num_threads)(1) });
}

/// `stdlib/blas/blas.cleave`'s own `Sgemm::sgemm` -- the *one* generic
/// binding `doc/plan-blas-native.md` §7.1 calls for, replacing the first
/// attempt's own four shape-specific wrappers. `trans_a`/`trans_b`: `0` for
/// `CBLAS_NO_TRANS`, `1` for `CBLAS_TRANS` (`stdlib/blas/blas.cleave`'s own
/// doc comment on why a plain `i32` crosses this boundary, not a `bool` --
/// cleave's own `bool` has no guaranteed C-ABI representation this crate
/// wants to depend on). `_a_len`/`_b_len`/`_c_len`: every array-typed cleave
/// argument crosses the extern boundary as a `(pointer, i64 length)` pair
/// (`mlir_lower.rs::array_ptr_and_len`'s own doc comment), even though the
/// length is already compile-time-known on the cleave side -- unread here,
/// same as the first attempt's own four wrappers. `lda`/`ldb`/`ldc`: row-
/// major leading dimension is always the operand's own *physical* trailing
/// extent, regardless of `trans_a`/`trans_b` -- `stdlib/blas/blas.cleave`'s
/// own call site computes and passes these explicitly, this wrapper trusts
/// them unconditionally, exactly like every other dimension here.
///
/// # Safety
/// `a` must point to at least `lda * (if trans_a != 0 { k } else { m })`
/// valid, initialized `f32`s; `b` similarly for `lda`/`m`/`k` replaced by
/// `ldb`/`k`/`n`; `c` to at least `ldc * m` valid `f32`s, writable. All
/// three non-overlapping. Entirely the caller's own contract to uphold --
/// this is `doc/plan-blas-native.md` §7's own deliberately low-level,
/// unchecked entry point, mirroring the real `cblas_sgemm` C ABI directly.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cleave_blas_sgemm(
    trans_a: i32,
    trans_b: i32,
    m: i32,
    n: i32,
    k: i32,
    alpha: f32,
    a: *const f32,
    _a_len: i64,
    lda: i32,
    b: *const f32,
    _b_len: i64,
    ldb: i32,
    beta: f32,
    c: *mut f32,
    _c_len: i64,
    ldc: i32,
) {
    ensure_thread_count_pinned();
    unsafe {
        (blas_dynload::fns().cblas_sgemm)(
            CBLAS_ROW_MAJOR,
            if trans_a != 0 { CBLAS_TRANS } else { CBLAS_NO_TRANS },
            if trans_b != 0 { CBLAS_TRANS } else { CBLAS_NO_TRANS },
            m,
            n,
            k,
            alpha,
            a,
            lda,
            b,
            ldb,
            beta,
            c,
            ldc,
        );
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
            // the count" check was tried here and removed at first, not
            // left red: it asserted a fresh allocation reuses the
            // just-freed address, found directly to be unreliable against
            // the real system allocator (Windows' own allocator doesn't
            // guarantee immediate reuse the way some allocators' fast
            // paths do). **Now genuinely testable, the size-class free-list
            // cache's own doc comment (`FREE_LISTS`, above) having made
            // address reuse an actual, deterministic contract of this
            // allocator rather than an implementation detail of whichever
            // OS allocator happens to sit underneath it** — a released
            // block of a given size class is *always* the next block
            // handed back to a same-class request, LIFO, with no OS call
            // in between at all.
            let p1 = cleave_alloc_rc(40);
            cleave_release(p1);
            let p2 = cleave_alloc_rc(40);
            assert_eq!(
                p1, p2,
                "a released block must be reused by the very next same-size-class allocation"
            );

            // Two *simultaneously live* same-class allocations must still
            // never alias — the cache only ever hands out a block once
            // it's actually been released, exactly like the plain
            // `std::alloc::alloc` path it replaces.
            let live_a = cleave_alloc_rc(40);
            let live_b = cleave_alloc_rc(40);
            assert_ne!(live_a, live_b, "two live allocations must not overlap, cache or no cache");
            cleave_release(live_a);
            cleave_release(live_b);

            // A smaller request that still rounds up to the *same* size
            // class reuses the identical cached block too — the free list
            // is keyed by class, not by exact `data_size` (40 and 38 bytes
            // of data, 56 and 54 with the header, are one class of 56).
            let p3 = cleave_alloc_rc(40);
            cleave_release(p3);
            let p4 = cleave_alloc_rc(38);
            assert_eq!(
                p3, p4,
                "a smaller same-class request must still reuse the previously released block"
            );
            cleave_release(p4);

            // The cached block is real, writable memory each time it comes
            // back around — not just an address that happens to satisfy
            // the assertions above. Cycles through the class' own free
            // list several times, writing a different pattern each time.
            let mut prev: *mut i64 = std::ptr::null_mut();
            for i in 0..5i64 {
                let p = cleave_alloc_rc(40) as *mut i64;
                if !prev.is_null() {
                    assert_eq!(p as *mut i64, prev, "the free list should keep recycling this same block");
                }
                *p = 0x1000 + i;
                assert_eq!(*p, 0x1000 + i);
                prev = p;
                cleave_release(p as *mut u8);
            }

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

        // A small payload stays 16-byte aligned; one of at least
        // `VECTOR_ALIGN` bytes gets a 64-byte aligned data pointer, even
        // right after an odd-sized allocation left the cursor unaligned.
        let h3 = cleave_region_enter(4096);
        let p = cleave_alloc_local(h3, 17) as usize; // an odd size on purpose
        assert_eq!(p % 16, 0, "alloc_local's own result must be 16-byte aligned");
        let q = cleave_alloc_local(h3, 256);
        assert_eq!(q as usize % VECTOR_ALIGN, 0, "a large alloc_local payload must be 64-byte aligned");
        assert_eq!(unsafe { rc_count(q) }, 1, "the header must still sit right before the data");
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

    // `doc/plan-affine-ownership.md`'s Stage 2: `cleave_alloc_pool`/
    // `cleave_release_pool` -- no header, no refcount, `data_size` passed
    // again at release since nothing stores it.
    #[test]
    fn pool_alloc_writes_and_reads_back_correctly_with_no_header() {
        unsafe {
            let ptr = cleave_alloc_pool(8);
            assert!(!ptr.is_null());
            // With no header at all, the returned pointer is the true
            // base of the allocation -- writing/reading its own first
            // byte must never touch anything but this block's own
            // storage (there is no header slot in front of it to
            // accidentally corrupt, unlike `cleave_alloc_rc`).
            *ptr = 0x42;
            assert_eq!(*ptr, 0x42);
            *ptr.add(7) = 0x99;
            assert_eq!(*ptr.add(7), 0x99);
            cleave_release_pool(ptr, 8);
        }
    }

    #[test]
    fn a_size_class_holds_its_request_and_wastes_at_most_an_eighth() {
        let mut previous = 0;
        for n in (32..1 << 16).chain([1572928, 4194304, 4194368, 16777280, 1 << 40]) {
            let class = super::size_class(n);
            let bytes = super::class_bytes(class);
            assert!(class < super::NUM_SIZE_CLASSES);
            assert!(bytes >= n, "{n} bytes in a class of {bytes}");
            assert!(bytes - n <= n / 8, "{n} bytes in a class of {bytes}");
            assert_eq!(super::size_class(bytes), class, "{bytes}: a class's own size is in it");
            if n < 1 << 16 {
                assert!(class >= previous, "classes grow with the size");
                previous = class;
            }
        }
        assert_eq!(super::class_bytes(super::size_class(4194304)), 4194304);
        assert_eq!(super::class_bytes(super::size_class(4194368)), 4718592);
    }

    #[test]
    fn pool_release_and_alloc_reuse_the_same_block_for_the_same_size_class() {
        unsafe {
            let a = cleave_alloc_pool(16);
            cleave_release_pool(a, 16);
            let b = cleave_alloc_pool(16);
            assert_eq!(
                a, b,
                "a released block must be handed back out again for the \
                 next same-class allocation, exactly like `cleave_alloc_rc`'s \
                 own free-list -- confirms `cleave_alloc_pool`/`cleave_release_\
                 pool` genuinely share `FREE_LISTS`, not a separate pool"
            );
            cleave_release_pool(b, 16);
        }
    }

    #[test]
    fn large_payloads_are_64_byte_aligned_small_ones_stay_compact() {
        unsafe {
            for size in [64i64, 100, 4096, 401_408 * 4] {
                let p = cleave_alloc_rc(size);
                assert_eq!(p as usize % VECTOR_ALIGN, 0, "alloc_rc({size}) must be 64-byte aligned");
                assert_eq!(rc_count(p), 1);
                cleave_retain(p);
                assert!(!cleave_release(p));
                assert!(cleave_release(p), "alloc_rc({size}) must free cleanly from its padded base");
                // The freed block, handed back out headerless, is aligned too.
                let q = cleave_alloc_pool(size);
                assert_eq!(q as usize % VECTOR_ALIGN, 0, "alloc_pool({size}) must be 64-byte aligned");
                cleave_release_pool(q, size);
            }
            let small = cleave_alloc_rc(24);
            assert_eq!(small as usize % 16, 0);
            assert_eq!(rc_count(small), 1);
            assert!(cleave_release(small));
        }
    }

    #[test]
    fn pool_and_rc_allocations_freely_interchange_the_same_size_class_blocks() {
        // The exact claim `cleave_alloc_pool`'s own doc comment makes:
        // a block released headerless can be immediately reused by a
        // *headered* allocation of the same class, and vice versa --
        // proof that sharing one `FREE_LISTS` between the two allocator
        // pairs is safe, not just convenient.
        unsafe {
            let headerless = cleave_alloc_pool(8);
            cleave_release_pool(headerless, 8);
            // `cleave_alloc_rc(8)`'s own total (header + 8 bytes) may or
            // may not land in the identical size class as a bare 8-byte
            // pool request -- assert the *reachable* property instead of
            // exact pointer equality: the headered allocation must still
            // work correctly (real header, refcount 1, releasable) even
            // when it happens to reuse a block a headerless release just
            // returned.
            let headered = cleave_alloc_rc(8);
            assert_eq!(rc_count(headered), 1);
            assert!(cleave_release(headered));

            let headered2 = cleave_alloc_rc(8);
            assert!(cleave_release(headered2));
            let headerless2 = cleave_alloc_pool(8);
            *headerless2 = 7;
            assert_eq!(*headerless2, 7);
            cleave_release_pool(headerless2, 8);
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
// `DynArray<S>` of structs (`RawBuffer<S: HeapStruct>`): its slots hold
// references, like an array's. An extern's struct argument is lent and its
// struct result owned by the caller (`refcount.rs`), so `set` retains what it
// stores and releases what it overwrites, and `get` retains what it hands
// out. Without, the buffer held no reference to its elements: a point pushed
// from an array (`examples/convex_hull.cleave`) was freed with the array,
// then read back and released a second time. New slots are zeroed, an empty
// slot telling itself from an element. The elements still held when the
// `DynArray` dies are not released (its envelope has no cascade into the
// buffer): a leak, not a dangling reference.

#[unsafe(no_mangle)]
pub extern "C" fn dynarray_alloc_ptr(cap: i32) -> *mut *mut u8 {
    let bytes = cap as i64 * std::mem::size_of::<*mut u8>() as i64;
    unsafe {
        let p = cleave_realloc(std::ptr::null_mut(), 0, bytes);
        std::ptr::write_bytes(p, 0, bytes as usize);
        p as *mut *mut u8
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dynarray_grow_ptr(old: *mut *mut u8, old_cap: i32, new_cap: i32) -> *mut *mut u8 {
    let size = std::mem::size_of::<*mut u8>() as i64;
    unsafe {
        let p = cleave_realloc(old as *mut u8, old_cap as i64 * size, new_cap as i64 * size);
        if new_cap > old_cap {
            std::ptr::write_bytes(p.add((old_cap as i64 * size) as usize), 0, ((new_cap - old_cap) as i64 * size) as usize);
        }
        p as *mut *mut u8
    }
}

/// # Safety
/// `buf` must point to a live buffer of at least `i + 1` elements, slot `i`
/// holding an element (`dynarray_set_ptr`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dynarray_get_ptr(buf: *const *mut u8, i: i32) -> *mut u8 {
    unsafe {
        let v = *buf.add(i as usize);
        cleave_retain(v);
        v
    }
}

/// # Safety
/// `buf` must point to a live buffer of at least `i + 1` elements, and `v`
/// to a live struct.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dynarray_set_ptr(buf: *mut *mut u8, i: i32, v: *mut u8) {
    unsafe {
        cleave_retain(v);
        let slot = buf.add(i as usize);
        let old = *slot;
        *slot = v;
        if !old.is_null() {
            cleave_release(old);
        }
    }
}

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

/// The generator's whole state: `rand_seed(rand_state())` resumes the exact
/// sequence (what a checkpoint saves, `stdlib/checkpoint`).
#[unsafe(no_mangle)]
pub extern "C" fn rand_state() -> i64 {
    PCG_STATE.load(std::sync::atomic::Ordering::Relaxed) as i64
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

#[cfg(test)]
mod arena_thread_tests {
    use super::*;

    /// Regions opened and closed on several threads at once each keep their
    /// own allocations: a thread's `cleave_region_exit` rewinds its own arena,
    /// never another thread's (`ARENA`). With one global arena, a thread's
    /// exit rewound the shared cursor under the others' live blocks, and
    /// their next allocations overwrote them.
    #[test]
    fn regions_on_several_threads_dont_overwrite_each_other() {
        let threads: Vec<_> = (0..8u8)
            .map(|t| {
                std::thread::spawn(move || {
                    for round in 0..200u32 {
                        let outer = cleave_region_enter(0);
                        let a = cleave_alloc_local(outer, 4096);
                        unsafe { std::ptr::write_bytes(a, t, 4096) };
                        for _ in 0..4 {
                            let inner = cleave_region_enter(0);
                            let b = cleave_alloc_local(inner, 1024);
                            unsafe { std::ptr::write_bytes(b, t ^ 0xff, 1024) };
                            std::thread::yield_now();
                            cleave_region_exit(inner);
                        }
                        let intact = unsafe { std::slice::from_raw_parts(a, 4096) }.iter().all(|&x| x == t);
                        assert!(intact, "thread {t}, round {round}: a block was overwritten");
                        assert!(is_in_arena(a));
                        cleave_region_exit(outer);
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().expect("a thread failed");
        }
    }
}

#[cfg(test)]
mod pool_thread_tests {
    use super::*;

    /// The pool under several threads: each allocates blocks of mixed sizes,
    /// fills each with its own pattern, and hands half of them to the next
    /// thread to release — blocks migrate between thread caches and through
    /// the depot. A block handed out twice while live would show up as an
    /// overwritten pattern.
    #[test]
    fn the_pool_hands_out_each_block_once_across_threads() {
        const THREADS: usize = 8;
        let (senders, receivers): (Vec<_>, Vec<_>) =
            (0..THREADS).map(|_| std::sync::mpsc::channel::<(usize, u8, usize)>()).unzip();
        let handles: Vec<_> = receivers
            .into_iter()
            .enumerate()
            .map(|(t, rx)| {
                let next = senders[(t + 1) % THREADS].clone();
                std::thread::spawn(move || {
                    let mut live: Vec<(*mut u8, u8, usize)> = Vec::new();
                    for i in 0..4000usize {
                        let size = [24usize, 200, 3000, 70_000][i % 4];
                        let tag = ((t * 31 + i) % 251) as u8;
                        let p = cleave_alloc_rc(size as i64);
                        unsafe { std::ptr::write_bytes(p, tag, size) };
                        live.push((p, tag, size));
                        // Release some of this thread's blocks, give others away.
                        if live.len() > 64 {
                            let (q, qtag, qsize) = live.remove(i % live.len());
                            let ok = unsafe { std::slice::from_raw_parts(q, qsize) }.iter().all(|&x| x == qtag);
                            assert!(ok, "thread {t}: a live block was overwritten");
                            if i % 2 == 0 {
                                unsafe { cleave_release(q) };
                            } else {
                                next.send((q as usize, qtag, qsize)).unwrap();
                            }
                        }
                        // Release what the previous thread handed over.
                        while let Ok((q, qtag, qsize)) = rx.try_recv() {
                            let q = q as *mut u8;
                            let ok = unsafe { std::slice::from_raw_parts(q, qsize) }.iter().all(|&x| x == qtag);
                            assert!(ok, "thread {t}: a handed-over block was overwritten");
                            unsafe { cleave_release(q) };
                        }
                    }
                    for (q, _, _) in live {
                        unsafe { cleave_release(q) };
                    }
                    drop(next);
                    // Drain what is still in flight.
                    while let Ok((q, _, _)) = rx.recv_timeout(std::time::Duration::from_millis(200)) {
                        unsafe { cleave_release(q as *mut u8) };
                    }
                })
            })
            .collect();
        drop(senders);
        for h in handles {
            h.join().expect("a thread failed");
        }
    }
}

#[cfg(test)]
mod parallel_threads_tests {
    /// One thread per physical core (or `OMP_NUM_THREADS`): never more than
    /// the logical processors, at least one. On SMT hardware, half of them.
    #[test]
    fn parallel_threads_are_physical_cores() {
        let n = super::cleave_parallel_threads() as usize;
        let logical = std::thread::available_parallelism().unwrap().get();
        assert!(n >= 1 && n <= logical, "{n} threads for {logical} logical processors");
        if std::env::var_os("OMP_NUM_THREADS").is_none() {
            #[cfg(windows)]
            assert_eq!(Some(n), super::physical_cores());
        }
    }
}

#[cfg(all(test, windows))]
mod placement_tests {
    /// Each team member lands on its own physical core, all of that core's
    /// logical processors: never two on one core's SMT siblings.
    #[test]
    fn team_members_are_placed_on_distinct_physical_cores() {
        #[repr(C)]
        #[derive(Default)]
        struct GroupAffinity {
            mask: u64,
            group: u16,
            reserved: [u16; 3],
        }
        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn GetCurrentThread() -> *mut std::ffi::c_void;
            fn GetThreadGroupAffinity(thread: *mut std::ffi::c_void, affinity: *mut GroupAffinity) -> i32;
        }
        let cores = super::cores().expect("no core information");
        let team = cores.len().min(4);
        let placed: Vec<(u16, u64)> = (0..team)
            .map(|i| {
                std::thread::spawn(move || {
                    super::cleave_bind_worker(i as i32);
                    let mut affinity = GroupAffinity::default();
                    assert_ne!(unsafe { GetThreadGroupAffinity(GetCurrentThread(), &mut affinity) }, 0);
                    (affinity.group, affinity.mask)
                })
            })
            .map(|h| h.join().unwrap())
            .collect();
        for (i, &core) in placed.iter().enumerate() {
            assert_eq!(core, cores[i], "thread {i} isn't on core {i}");
        }
        let distinct: std::collections::HashSet<_> = placed.iter().collect();
        assert_eq!(distinct.len(), team, "two team members share a core: {placed:?}");
    }
}
