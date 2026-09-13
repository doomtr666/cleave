# Plan: region-scoped arena allocation for tensor temporaries

**Audience**: an agent picking this up cold, with no context from the session that produced it. Everything needed is here or precisely referenced. Read `doc/backlog.md`'s own "Region-scoped arena allocation..." entry first for the one-paragraph *why*; this file is the *how*.

**Goal**: make MLIR-bufferization-allocated tensor temporaries land in the per-iteration arena instead of the refcounted pool, so the long-standing CPS-vs-bufferization double-ownership bug class stops existing rather than being arbitrated per site.

> **Read §8 first.** It was written after Steps 0–3 were attempted and supersedes the ordering below: five distinct reconciliation criteria are now recorded as measured failures (§5), and §8 explains what that proves, which of the three possible exits to take, and the one cheap static check that must be answered before any of it is built. The arena (this file's original subject) remains the right end state, but as an optimization on top of unambiguous ownership — not as the mechanism that establishes it.

---

## 1. The problem, in one paragraph

`refcount.rs` (CPS level) counts **logical values**; MLIR's `--ownership-based-buffer-deallocation` counts **physical buffers**. The mapping between them is decided by One-Shot Bufferize *after* CPS has committed — one CPS value may get zero, one, or a shared buffer. So when a tensor value gets both an explicit `cleave_release` (from `refcount.rs`) and a `memref.dealloc` (from bufferization), the same memory is freed twice. Reconciling the two counts after the fact is provably heuristic: an ordinary singly-owned tensor and a genuinely double-owned one have the *identical* CPS-level signature (`releases == retains + 1`). Four distinct instances of this bug were found and patched one at a time before this direction was agreed; `compensate_refcounts.rs::suppress_bufferization_owned_releases` is that patch, currently shipping with a hardcoded shape list.

---

## 2. Map of what already exists (all verified against the real kernel, not assumed)

### Runtime — `cleave-rt/src/lib.rs`

| Thing | Where | Behaviour |
|---|---|---|
| `ARENA_BASE` / `ARENA_CURSOR` / `ARENA_CAPACITY` | ~line 673-698 | One contiguous 256 MB region, bump-pointer. |
| `cleave_region_enter(size) -> handle` | ~line 780 | Bumps `REGION_DEPTH`, returns the current cursor as the handle. |
| `cleave_region_exit(handle)` | ~line 855 | Rewinds `ARENA_CURSOR` straight back to `handle`, decrements `REGION_DEPTH`. Bulk reclaim, no per-object work. |
| `cleave_alloc_local(handle, size)` | ~line 836 | Bump-allocates in the arena. **Asserts** if `REGION_DEPTH == 0`. |
| `cleave_alloc_rc(size)` | ~line 381 | Always pool/heap. Never consults `REGION_DEPTH`. |
| `is_in_arena(ptr)` | ~line 752 | Address-range test against `ARENA_BASE`/`ARENA_CAPACITY`. |

### The single most important fact for this plan

**`cleave_release` already consults `is_in_arena` and performs no physical free for an arena-backed block** (`cleave-rt/src/lib.rs` ~line 606, inside the `refcount == 0` branch). It still returns `true` so a struct's own release cascade fires correctly.

Consequences, both load-bearing:
- **An arena-backed block is already immune to double-free.** Two decrements on it free nothing, ever. Whichever of the two mechanisms emits a release is irrelevant.
- **Releases do not need to be statically suppressed.** The existing design is already dynamic-safe: emit the release unconditionally, let `is_in_arena` decide at runtime. `cleave_release_void` (what `unify_alloc.rs` renames bufferization's `free` to) calls straight into `cleave_release`, so the bufferization path gets the identical treatment for free.

This is why the direction works: it does not require getting an ownership decision exactly right everywhere. It requires only getting the *allocator* right.

### Compiler — current wiring

- `region_analysis.rs::find_region_local_functions` returns the set of top-level function names safe to lower with `cleave_alloc_local`.
  - `analyze_loop_body` computes the **escape set** (`collect_escaping`: every `CVar` referenced in a tail call back to the same loop — i.e. what `scf.yield` carries) and marks a call region-local when `reaches_escaping(result_var, ...)` is false. **This analysis is correct and is exactly the CPS-level escape analysis this plan needs. Do not rewrite it.**
  - It is gated by `call_counts.get(callee) == 1` — exactly one call site in the whole program. **This gate is the real blocker; see §3.**
  - A worklist performs a transitive descent into already-marked callees (same `call_counts == 1` gate at every level).
- `mlir_lower.rs` line ~1462: `ctx.currently_region_local.set(ctx.region_local_fns.contains(&f.def.name))` — set **once per top-level function**.
- `alloc_llvm_value` (~line 4166) picks `cleave_alloc_local` vs `cleave_alloc_rc` from that flag. `tensor_seed` (~line 4890) does the same for tensor seeds.
- `lower_loop` (~line 2019) only emits `cleave_region_enter`/`exit` when some callee in the loop is region-local.

### Measured state of the real `mnist-interop` kernel

Taken from `CLEAVE_DUMP_POST_DEALLOC` output (see §6 for how to regenerate):

- Regions are already opened and closed **per epoch iteration and per batch iteration**, nested. The scoping is already exactly where it needs to be.
- **62** allocations already go through `cleave_alloc_local`.
- **153** still go through `memref.alloc` (bufferization's own), with **184** `memref.dealloc`.

Those 153 are the target population, and are exactly where every instance of the double-release bug has come from.

---

## 3. The blocker, and how to lift it

`call_counts == 1` exists because **the allocator decision is attached to a function, not to a call site** (`currently_region_local` is set once per function, above). If one monomorphized function is called from both a region-local context and a non-region-local one, its single lowered body would use `cleave_alloc_local` at *both*, and the non-region-local call site would hit `assert_region_open` and abort. The one-call-site rule is a blunt way to make that impossible.

This hits real code: shared monomorphized algebra functions (`MatMul::matmul<...>`, `Ring::add<...>`) called from both the training loop and evaluation are excluded outright. `doc/backlog.md` already carries this as its own entry ("`net_grad`'s own remaining internal tensor allocations still all go through `cleave_alloc_rc`").

### Recommended lift: function specialisation, not per-call-site analysis

Duplicate, don't contextualise. For each function that is region-local at *some* call sites and not at others, emit two CPS-level copies — e.g. `f` and `f$region` — and retarget each call site to the appropriate one. The existing per-function analysis and the existing per-function lowering flag then both stay exactly as they are.

Why this over a genuine per-call-site lowering mode:
- The escape analysis (`reaches_escaping`) needs no change at all — it already answers the question per call site; only the *consumer* of its answer was coarse.
- `monomorphize.rs` already duplicates functions per type shape, so cloning a `CFunDef` under a fresh name and rewriting call targets is an established, exercised operation in this codebase.
- Code-size growth is bounded (at most ×2 for affected functions) and largely evaporates: `--inline` runs afterwards in `pipeline.rs`.
- It is a pure CPS→CPS transform, testable in isolation without touching MLIR.

**Placement — check this before designing, it constrains the shape.** `find_region_local_functions` is called from *inside* `lower_program` (`mlir_lower.rs:378`), which receives an already-final `&CpsProgram`. Specialisation rewrites the program (new function definitions, retargeted call sites), so it cannot live there. It must run as a CPS→CPS pass **before** `lower_program` is called — i.e. in `pipeline.rs`'s CPS-building stage, alongside `insert_refcounting`, producing the split program that `lower_program` then consumes unchanged. `find_region_local_functions` then simply runs on the already-split program and marks the `$region` copies, with its internal logic untouched.

Note the call sites: `pipeline.rs:1664` plus three in `main.rs` (~480, ~528, ~622, the `--dump-*`/`--emit-*` paths). Put the specialisation where all of them pick it up — inside the shared CPS-building helper, not at one call site.

### Secondary hardening, independently valuable

Make `cleave_alloc_local` fall back to the pool instead of asserting when `REGION_DEPTH == 0`. Today a misclassification aborts the process; with the fallback it degrades to "allocated on the pool instead of the arena", and because releases are always emitted and `is_in_arena` decides dynamically, the block is then correctly pool-freed. This converts the entire class of classification errors from crashes into a perf regression — which makes every subsequent step in this plan far cheaper to iterate on.

Keep the assert available behind a debug env var so genuine compiler bugs stay visible.

---

## 3bis. Headerless arena allocations, and the alignment they unlock

Two separate wins share one mechanism. Read both before implementing either — the second is the larger one and it constrains the design of the first.

### Why the header can go, for the population this plan targets

`cleave_alloc_local` deliberately writes the same `RcHeader` as `cleave_alloc_rc` (`cleave-rt/src/lib.rs` ~798-808), because codegen emits the same retain/release either way and a headerless block would be dereferenced blindly. That constraint is real — `cleave_release` decrements `ptr - 16` (~line 596) *before* testing `is_in_arena` (~line 606) — but it is stated more strongly than the facts require. On an arena block the two header fields are used for:

- **`data_size`** — read only inside the `!is_in_arena` branch, to pick the free-list size class. **Dead for every arena block, unconditionally.**
- **`refcount`** — read only to produce the `i1` return value, whose sole consumer is `mlir_lower.rs::lower_release_cascade`, the descent into a struct's refcounted fields.

So the header is load-bearing, on the arena, for exactly one thing: **the struct field cascade**. Which gives the real criterion — not "temporary" but **leaf vs container**:

- A **leaf** (a flat tensor payload) has no refcounted fields, so no cascade, so nothing reads the return value. Its header is 100% dead.
- A **container** (a struct with rc fields) genuinely needs the count: it can legitimately reach 2 inside a region, and the cascade must fire exactly once.

**The population Step 3 targets is entirely leaves.** One-Shot Bufferize only ever allocates flat `memref`s — never a cleave struct. So all 153 sites are eligible by construction; no classification pass is needed to separate them.

### Do not make the discriminator dynamic

The tempting shortcut — put `if is_in_arena(ptr) { return true }` at the top of `cleave_release` — **is wrong**. It would fire the cascade on every release of an arena-backed container instead of once at zero. The discriminator must be "headerless", not "in the arena", and headerlessness is a static property.

The clean split exploits a guarantee that already exists:

- **`cleave_release`** keeps its current order. It can receive containers.
- **`cleave_release_void`** (`cleave-rt/src/lib.rs` ~883, today a thin wrapper) can safely take an `is_in_arena`-first early-out, because `unify_alloc.rs` emits it *only* as the rename of bufferization's `@free` — so it provably only ever sees bufferization buffers, i.e. leaves. The guarantee is structural, not an assumption. **This variant is independent of Steps 2-3 and can land on its own.**

Allocation side: add `cleave_alloc_local_raw(handle, size)` = a bare `arena_bump(size)` with no header write. And where the compiler knows both the type (leaf) and the allocator (region-local), the release is not lightened — **it is not emitted at all**; `cleave_region_exit` reclaims in bulk.

That last part rides on Step 2: the allocator must be known *at the release site*, which is the same per-function coarseness §3 lifts. Note also that a leaf arriving as a *parameter* of a region-local function may be pool-backed, so elision applies only to values whose allocation site is in the same function — decidable with `reaches_escaping` unchanged.

Structural payoff, beyond cycles: **a block with no refcount cannot be double-released.** This removes the bug class rather than neutralising it.

### Alignment — the larger win, and the reason to do this at all

**AVX-512 is genuinely in play here, confirmed, not assumed.** `pipeline.rs`'s `stamp_target_cpu` doc comment records a direct disassembly finding: the emitted object contains *hundreds* of `zmm` instructions even when built with `--target-cpu x86-64-v2`, because `melior::ExecutionEngine` always builds its own `TargetMachine` from `JITTargetMachineBuilder::detectHost()`. So the kernel is already being vectorised at 512 bits on this Zen host, regardless of the (currently inert) flag.

Now the arithmetic. `RC_HEADER_SIZE` is 16. `arena_base()` is 64-aligned (`Layout::from_size_align(ARENA_CAPACITY, 64)`), `arena_bump` rounds the cursor to 16, and the payload sits at `base + 16`. `cleave_alloc_rc` is the same shape with `from_size_align(total, 16)`. So **every tensor payload in the system is 16-aligned and, relative to the 64-byte vector width, misaligned by construction.** A full-width 64-byte access at payload offset 0 straddles a cache line. That is not a rounding detail — it is a guaranteed line-split on a large fraction of accesses in a matmul-heavy kernel.

The header is therefore not costing 16 bytes (0.02% of a `32x512xf32`). **It is what forces the misalignment.**

Two caveats that decide how to spend effort here, both of which must be respected or the work will measure as a no-op:

1. **`vmovups` on aligned data is not slower than `vmovaps` on Zen or on any Intel core since Nehalem.** The instruction form is not the problem. The cost is the *actual* line-splitting, plus load-queue pressure. So do not chase an instruction-mix change as the goal.
2. **LLVM must be *told* the alignment, or aligning the data changes nothing.** Today it cannot know: `mlir_lower.rs` sets `allocated_ptr` and `aligned_ptr` to the same fresh pointer with no rounding (~3215-3217), and the `llvm.func` declarations for `cleave_alloc_rc`/`cleave_alloc_local` carry no `align` return attribute. The `arena_bump` doc comment's own observation — that payload accesses come out as `vmovups` and `llvm.intr.masked.load` — is the symptom of exactly this: an unknown-alignment vectoriser emitting masked/peeled forms rather than clean full-width ones. **The vectoriser's conservatism is probably worth more than the line splits.**

Concretely, three changes, and the ordering matters:

- **(a) Align the data.** For headerless arena leaves this is free: payload *is* the bump address, so round the bump to 64 instead of 16. For blocks that keep a header (pool, arena containers), place the payload on a 64 boundary with the header in the 16 bytes immediately below it — `rc_header(ptr) = ptr - 16` stays byte-for-byte valid. **Wrinkle to handle, not skip:** `cleave_release` pushes `header as *mut u8` onto the free list and `cleave_alloc_rc` pops it back as the block base. If the payload is shifted, that recovered base is no longer the pointer `std::alloc::alloc` returned. Making the shift deterministic per size class keeps reuse self-consistent, but the `class >= NUM_SIZE_CLASSES` fallback genuinely calls `std::alloc::dealloc(base, layout)` and would then pass a shifted pointer — unreachable in practice (`total > 2^63`) but it must not be left incoherent.
- **(b) Declare it.** Either an `align 64` return attribute on the allocator `llvm.func` declarations (via `ensure_extern_declared`, `mlir_lower.rs:5918`), or the idiomatic MLIR route: emit `memref.assume_alignment %m, 64` right after the descriptor is materialised. This is the single highest-leverage change of the three and the cheapest to try first.
- **(c) Measure the vectoriser, not the vibes.** Before/after counts of `llvm.intr.masked.load`/`masked.store` in the `CLEAVE_DUMP_POST_DEALLOC` dump, and of `vmovups`/`vmovaps`/`vmovapd` in the disassembled object.

**Ordering trap:** doing (a) without (b) will measure as a no-op and may be wrongly concluded to be worthless. Do (b) first on the *current* 16-aligned layout — it is a one-line-ish change and it will either move the instruction mix or prove LLVM is already inferring something. Only then does (a) have a measurable baseline to improve on.

Hand the wall-clock comparison to the user rather than timing it here; this repo's own convention (`doc/backlog.md`'s OpenMP sweep) is 10 epochs of `mnist-interop`, ~12-19 s baseline.

---

## 4. Ordered steps

### Step 0 — reproduce the baseline

```
cargo build --release -p mnist-interop
CLEAVE_DEBUG_POOL=1 ./target/release/mnist-interop.exe
```
Expect: a `CLEAVE_DEBUG_POOL: cleave_release on parked (already-freed) block ... data_size=104` crash, early, before any `Epoch=` line. That is the currently-open 4th instance (network-construction time, mechanism not yet diagnosed — see §7).

Regenerate the allocation census:
```
CLEAVE_DUMP_POST_DEALLOC=/tmp/post.mlir ./target/release/cleave.exe examples/mnist-interop/src/kernel.cleave --emit-object /tmp/k.o
grep -c "memref.alloc()" /tmp/post.mlir      # raw count includes decoys, see below
grep -c "cleave_alloc_local" /tmp/post.mlir  # expect 62
grep -c "memref.dealloc" /tmp/post.mlir      # expect 184
```

**Reconciliation, confirmed by direct re-run, not assumed**: the raw `memref.alloc()` grep currently reads **177**, not ~153. The gap is exactly accounted for: `compensate_refcounts.rs::patch_deallocs` inserts a **fresh dummy `memref::alloc()`** in front of every dealloc it neutralises (its own doc comment: "inserts a fresh `memref::alloc()` before the dealloc... never erases"), and `CLEAVE_TRACE_SUPPRESS=1` on the same build reports exactly **24 dealloc(s) redirected**. `177 - 24 = 153`, matching this plan's number precisely. So the real Step-3 target population is `grep -c "memref.alloc()" − <redirect count from CLEAVE_TRACE_SUPPRESS=1>`, not the raw grep. Re-run `CLEAVE_TRACE_SUPPRESS=1` alongside the census at every later step for this reason — the decoy count moves as `suppress_bufferization_owned_releases`'s hardcoded shape list keeps matching until Step 4 removes it.

### Step 1 — `cleave_alloc_local` pool fallback (hardening, no behaviour change expected)

`cleave-rt/src/lib.rs::cleave_alloc_local`: when `REGION_DEPTH == 0`, allocate via the same path `cleave_alloc_rc` uses instead of asserting.

Validate: full suite green, `mnist-interop` unchanged (same 104-byte crash, no new leak). This step should be a no-op on correct code — its value is that every later step now fails soft.

### Step 1bis — declare allocator alignment to LLVM (independent, cheap, do it early)

Change (b) of §3bis, on the *current* 16-aligned layout: `align 64` return attribute on the allocator declarations, or `memref.assume_alignment`. Independent of every other step.

Expect one of two outcomes, both informative: the masked-load/`vmovups` mix shifts (alignment was the vectoriser's blocker — §3bis's (a) is then worth real effort), or nothing moves (LLVM was already inferring, and (a) buys only the line splits). Note the declaration is a *lie* until (a) lands — 16-aligned data declared as 64-aligned is UB, so treat this as a measurement probe on a scratch branch, not something to ship on its own.

Also independent and shippable on its own: the `cleave_release_void` arena-first early-out (§3bis).

### Step 2 — function specialisation to lift `call_counts == 1` — DONE

Implemented as two independently-tested pieces:

- **2a**: `region_analysis.rs` rewritten around a single whole-program fixed point (`analyze`), replacing `call_counts == 1` with the sound general condition — a callee is region-local iff *every* call site targeting it, anywhere in the program, is proven safe (`safe_sites`, a `HashSet<CVar>` of call-site identities, not a numeric tally — the module's own doc comment explains why identity, not counting, is what keeps transitive propagation sound). `find_region_local_functions`'s own public signature is unchanged. 12 tests in `cleave/tests/region_analysis.rs` (3 new, 1 updated to reflect the now-intentionally-relaxed verdict — see that test's own doc comment for why `shared_helper`, called twice from the same loop, is correctly *not* excluded any more).
- **2c**: `region_specialize.rs` (new module) — `specialize_region_local_functions`, the actual duplication for a callee genuinely split between a safe and an unsafe call site (`{name}$region`, `CVar`s renumbered via the same `FreshVars::starting_at(max_cvar_in_program(...) + 1)` idiom `refcount::insert_refcounting` already uses). Wired into `pipeline.rs::build_optimized_cps` and all three `main.rs` CPS-building sequences (`--dump-cps-optimized`, `--dump-mlir`/`--dump-mlir-lowered`, `--run`), always immediately before `region_analysis::find_region_local_functions` is ever consulted. 4 tests in `cleave/tests/region_specialize.rs`.

**Measured on the real kernel, before vs. after (`CLEAVE_TRACE_REGION_LOCAL=1`, `mlir_lower.rs`, kept as a permanent diagnostic)**: `region_local_fns` grew from 67 to 75 names, a **strict superset — zero names lost**, confirming the relaxation is sound in practice, not just in the unit tests. The 8 additions: 4 genuinely new "wholesale safe" functions from 2a alone (`Index::index<Tensor<f32,1,10>,f32>`, `Rand::normal<f32>`, `Ring::add<Tensor<f32,32,10>>`, `Ring::mul<f32>`), plus 4 real `$region` splits from 2c (`Ord::lt<i32>`, `Ring::add<f32>`, `Ring::add<i32>`, `Ring::zero<f32>` — all loop-counter/accumulator primitives, shared between a training loop's own safe use and some other, unsafe call site).

**Important, honest finding, not the outcome originally expected**: `MatMul::matmul<...>`/`MatMulTransposeA`/`MatMulTransposeB`/`dense_forward`/`net_grad` — the tensor-heavy functions this whole plan is really about — were **already** in `region_local_fns` *before* Step 2 (confirmed by the same before/after trace: all present in the 67-name baseline). Each monomorphized shape genuinely does have exactly one call site in this program, exactly as `region_analysis.rs`'s own pre-Step-2 doc comment already predicted ("each monomorphized instantiation really does have exactly one call site in a real network"). So Step 2 was never the blocker for *this* population. Correspondingly, the `cleave_alloc_local`/`memref.alloc` census barely moved (`cleave_alloc_local` 62 → 64; `memref.alloc`'s real target population unchanged at 153) — expected, not a sign Step 2 under-delivered: `region_local_fns` only feeds `mlir_lower.rs`'s own CPS-level `alloc_llvm_value`/`tensor_seed` choice today. **Nothing yet routes bufferization's own `memref.alloc` calls through it at all — that link is Step 3 itself**, not a byproduct of Step 2. Step 3 can now proceed against a `region_local_fns` that is strictly larger and provably sound, but should not expect the 153-population to have already shrunk on its own.

Validated: `cargo test -p cleave --release --no-fail-fast` green throughout (805 → 808 after 2a → 812 after 2c). `mnist-interop`'s own pre-existing, unrelated `data_size=104` crash (§7) reproduced byte-for-byte identical after every sub-step — confirms Step 2 touches nothing relevant to it.

### Step 3 — route bufferization's own allocations into the arena — ATTEMPTED, REVERTED, real leak found on the actual kernel

**Do not touch the tensor/memref level** — see §5 for why that has already failed once. **Do not re-attempt the specific "(b) Dynamic, module-wide, `REGION_DEPTH`-only" design described below without first fixing the region-scope mismatch this entry documents** — it passed every unit test in `cleave/tests/unify_alloc.rs` and still leaked multiple GB in under 10 seconds on the real `mnist-interop` kernel. This is exactly the class of failure §6's validation protocol exists to catch, and it did.

**What was built**: `cleave_rt::cleave_alloc_auto(size)`, a single-argument entry point (matching `malloc`'s own arity — `cleave_alloc_local`'s real `(handle, size)` can't be targeted by a bare callee rename) that checks `REGION_DEPTH` at the moment it runs and delegates to `cleave_alloc_local(0, size)` — arena-backed if some region is open, pool-backed otherwise (Step 1's own fallback). `unify_alloc.rs::unify_tensor_allocations` renamed every `llvm.call @malloc` module-wide to `@cleave_alloc_auto`, except one structurally nested inside a surviving `omp.parallel`/`omp.wsloop`/`omp.loop_nest`/`scf.parallel` region (left on `@cleave_alloc_rc`, per §7's thread-safety reasoning — this specific exclusion is still believed correct and is the one part of this attempt worth keeping conceptually).

**Why the *first* draft's `region_local_fns`-gating was dropped, and why that specific reasoning was itself correct.** An early version additionally gated the rename on "is the `malloc` call's *enclosing* `llvm.func`'s own name in `region_local_fns`". A real end-to-end test failed immediately: `--inline` (run long before this stage) had already spliced the region-local callee's own body into its caller, so "the enclosing `llvm.func`'s name" was the *outer*, non-region-local orchestrating function — the check silently matched nothing. Dropping it and reasoning that `REGION_DEPTH > 0` alone should be sufficient (`cleave_region_enter`/`cleave_region_exit` bracket *some* proven-safe dynamic extent, and CPS's sequential discipline means nothing else executes in between) made every unit test in `cleave/tests/unify_alloc.rs` pass — but those tests only ever exercise a loop with exactly *one* callee inside it. **The real kernel's own loop bodies don't look like that.**

**Root cause, confirmed directly in `mlir_lower.rs::lower_loop`'s own code comment, not guessed**: the region opened for a loop iteration spans the *entire* iteration body, deliberately — *"one region per loop iteration, opened here at the very start of the body, closed... right before this same iteration's own tail-call — spanning the whole iteration, not just an individual call within it, because a region-local function's own result can genuinely need to stay valid past its own call returning (`net_grad`'s own `g.2`, read afterward by `Optimizer::step`, is exactly this shape)."* So a training-loop iteration wraps **both** `net_grad`'s call (region-local, safe) **and** `Optimizer::step`'s call (**not** region-local — its own result, `net`/`state`, is exactly the loop's own carried, escaping state) in the *same* open region. Before this step, that was harmless: `alloc_llvm_value`'s own allocator choice is decided *statically, per function, at CPS-lowering time* — that function's own doc comment says so explicitly ("genuinely doesn't care whether a region happens to be open around it") — so `Optimizer::step`'s own internal allocations *always* used `cleave_alloc_rc`, unconditionally, regardless of `REGION_DEPTH`. This step's module-wide, purely-dynamic rename broke that invariant: `Optimizer::step`'s own free-standing tensor intermediates (real, weight-shaped tensors — large) now *also* saw `REGION_DEPTH > 0` and got arena-backed, for no reason tied to their own actual safety.

**What is confirmed versus still genuinely open**: that this specific invariant break is real and reproducible is confirmed (the code comment above, and the fact that reverting removes the leak entirely — same known, pre-existing, unrelated `data_size=104` crash returns, byte for byte, no growth beforehand). *Not* fully pinned down: the exact mechanism connecting "some allocations that used to be pool-backed are now arena-backed" to a multi-GB-in-seconds leak specifically (as opposed to a crash or silently wrong numbers) — a real candidate, not yet proven, is that bufferization's own deallocation point for a given buffer (decided by `--ownership-based-buffer-deallocation`'s own liveness analysis, entirely independent of where CPS-level `cleave_region_enter`/`exit` calls sit) can be scheduled *after* the enclosing iteration's own `cleave_region_exit` — meaning the arena has already rewound and handed that address to a *later* allocation by the time the original's matching `cleave_release_void` finally runs, corrupting whatever now occupies that memory rather than freeing it. Confirm or rule this out with real evidence (`CLEAVE_DUMP_POST_DEALLOC`, tracing one specific arena address's own allocate/free order against its enclosing region's own enter/exit) before trying again — do not re-guess.

**Reverted in full**: `cleave_rt::cleave_alloc_auto` removed, its `register_cleave_rt_symbols` entry removed, `unify_alloc.rs` and `cleave/tests/unify_alloc.rs` restored to their pre-Step-3 state (`git checkout`, confirmed clean — neither file carried any Step 1/2 changes). Verified after reverting: full suite green, `mnist-interop` back to the identical known baseline crash (§7's own 4th-instance entry), no leak under the same memory-monitoring protocol that caught the problem.

**Before re-attempting**, the real fix needs a way to know, *per allocation site, surviving `--inline`*, whether that specific site originated inside a function `region_analysis.rs` actually proved region-local — not "is `REGION_DEPTH > 0` right now" (too coarse, as shown above) and not "what's this op's enclosing `llvm.func` called *now*" (wrong after inlining, as shown even earlier). The most promising direction not yet tried: stamp a discardable MLIR attribute (e.g. `cleave.region_local_body`) on every operation inside a `region_local_fns` member's own body, at CPS-lowering time, *before* `--inline` ever runs — MLIR's generic inliner clones individual operations (and their own attributes) when splicing a callee's body into a caller, so a per-operation marker (not a per-function one) should survive being spliced elsewhere, unlike the function-name check that didn't. This needs its own careful, incremental validation against the real kernel — not just unit tests shaped like the ones that already missed this — before being trusted.

Do not mark this step done until a redesign passes the *same* memory-monitoring protocol this attempt failed, on the real kernel, for at least 2 epochs.

### Step 3bis — headerless arena leaves, 64-byte aligned

With Step 3's population in the arena, apply §3bis: `cleave_alloc_local_raw` (no header write), bump rounded to 64 so the payload is natively 64-aligned, and the `cleave_release_void` early-out covering the bufferization-emitted frees. Then make Step 1bis's alignment declaration truthful and re-measure.

Validate: census, full suite, memory monitoring, accuracy unchanged (an alignment or headerless bug shows up as **wrong numbers**, so the accuracy check is the load-bearing one here, not the crash check). Then hand the user the 10-epoch `mnist-interop` timing command for the wall-clock comparison.

### Step 4 — retire the per-site patch

Once Step 3 covers the population, `compensate_refcounts.rs::suppress_bufferization_owned_releases` and its hardcoded shape list (`memref<32x512xf32>` etc.) should become dead. Confirm by setting `CLEAVE_TRACE_SUPPRESS=1` and checking the redirect count falls to 0, then delete the function, its call site in `pipeline.rs`, and the shape list. Leave `compensate_merged_refcounts` alone — it addresses a genuinely different bug (CSE-merged allocations).

---

## 5. Things already tried that failed — do not repeat

| Attempt | Result |
|---|---|
| Skip `--ownership-based-buffer-deallocation` entirely, let `refcount.rs` own everything | **12.7 GB resident in 10 s.** `refcount.rs` does not track most free-standing intermediate tensor arithmetic; bufferization genuinely owns it today. |
| Remove `is_bare_tensor_ty` from `RefcountCtx::is_rc`, let MLIR own everything | Reintroduces an already-fixed leak; `a_bare_tensor_returned_from_a_call_and_consumed_only_as_a_borrowed_argument_is_released` fails immediately. |
| Arena-back tensor storage by building a hand-rolled memref descriptor + `unrealized_conversion_cast` around `cleave_alloc_local` at the **tensor/memref level** | ~63 s of extra `memcpy` traffic. The buffer became opaque to One-Shot Bufferize's alias analysis, which then lost its ability to elide copies. **This is why Step 3 works downstream of bufferization instead.** See `mlir_lower.rs::lower_tagged_struct_construct`'s doc comment. |
| General "release surplus" criterion in `suppress_bufferization_owned_releases` (`releases > retains`, no locality) | ~22 GB leak over 2 epochs. |
| Same, plus same-effective-scope locality | ~3.6 GB in 3-6 s. Still leaks with weight-gradient shapes excluded too — the false-positive class was never isolated. |
| Step 3, module-wide `malloc` → `cleave_alloc_auto`(size)` rename, gated purely on runtime `REGION_DEPTH > 0`, no per-function/per-site static check at all | Multi-GB growth in under 10 s on the real kernel, despite 5/5 green unit tests. Root cause: a loop iteration's own region spans the *whole* iteration (`mlir_lower.rs::lower_loop`'s own doc comment), not just the one region-local call within it — a co-resident, genuinely non-region-local call (`Optimizer::step`, called in the same iteration as `net_grad`) saw `REGION_DEPTH > 0` too and got its own free-standing allocations arena-backed, breaking the static, per-function allocator choice that made this safe pre-Step-3. Reverted in full — see Step 3's own entry above for the complete writeup and what to try instead. |
| Extend `suppress_bufferization_owned_releases`'s shape list with the four `Dense::b` bias shapes (`memref<1x{512,256,128,10}xf32>`) | Crash gone, **~10 GB in 48 s**. Those buffers are allocated once per *batch* while the matching CPS release cascade fires once per *epoch*, so all but roughly one of the ~1875 suppressed deallocs per epoch were genuinely needed. |
| `cleave_retain` on `dps_rewrite::Strategy::Overwrite`'s own redirected destination, symmetric with what `Strategy::Passthrough` already emits | Crash unchanged. 16 retains really were emitted, but the crashing block received **zero** — it is not an `Overwrite` destination at all. |
| Runtime provenance bit in `RcHeader` (`cleave_alloc_buf` marks bufferization's own allocations; `cleave_release` skips them — "whoever allocated it frees it") | Crash gone, **~9 GB in 36 s**, with *and* without the suppression pass. The rule is simply false: bufferization deliberately does *not* deallocate buffers whose ownership it hands to the caller, so blanket-skipping the CPS release leaks exactly those. |
| Neutralize the *CPS* release instead (rename `cleave_release` → `cleave_release_noop`) whenever the same buffer provably already has a `memref.dealloc` | Crash gone, **~8.9 GB in 40 s**. Neutralized 43 releases where the older shape list redirected 24; the surplus are releases that were the buffer's only real owner. |

---

## 6. Validation protocol — mandatory

**Always monitor real memory footprint.** Both leaks above were fast (seconds), and a first poll within 10 s would have caught either. Never conclude "it works" from a crash-free short run alone.

```bash
./target/release/mnist-interop.exe > /tmp/run.txt 2>&1 &
for i in $(seq 1 10); do
  sleep 3
  mem=$(powershell -NoProfile -Command "(Get-Process mnist-interop -ErrorAction SilentlyContinue).WorkingSet64")
  [ -z "$mem" ] && { echo "exited"; break; }
  mb=$((mem/1024/1024)); echo "t=$((i*3))s: ${mb} MB"
  [ "$mb" -gt 2000 ] && { powershell -NoProfile -Command "Stop-Process -Name mnist-interop -Force"; echo KILLED; break; }
done
```

Healthy: flat or slowly-oscillating working set. Any monotonic climb past ~1 GB in the first 30 s is a leak — revert and bisect.

Also required each step:
- `cargo test -p cleave --release --no-fail-fast` fully green. Note: the `pipeline` and `mlir_lower` targets are **intermittently flaky under parallel load** (link/JIT contention). Re-run the failing target in isolation before treating it as a real failure.
- Build env: `MLIR_SYS_220_PREFIX=/i/Dev/llvm-mlir-22 TABLEGEN_220_PREFIX=/i/Dev/llvm-mlir-22` must be set inline on every cargo/cleave.exe invocation; they are not in the ambient environment.
- Accuracy unchanged where a run completes: `digits-interop` `0.94713414`, `mnist-interop` `0.9342`.

### Debugging tools available

- `CLEAVE_DEBUG_POOL=1` — crashes deterministically with `int3` on a double-release, printing block address, refcount, `data_size` and tag.
- `CLEAVE_DUMP_POST_DEALLOC=<path>` — module right after buffer-deallocation, memref level, before `--convert-to-llvm` makes everything opaque. The only place `memref.dealloc`, its `scf.if` ownership wrapper, and `dealloc_helper` calls are readable.
- `CLEAVE_DUMP_PRE_BUFFERIZE=<path>` — still tensor-typed, before One-Shot Bufferize.
- `CLEAVE_TRACE_SUPPRESS=1` — every `suppress_bufferization_owned_releases` candidate with its shape and match status.
- `CLEAVE_TRACE_DEDUP=1` — every `wrap_releases` entry with its resolved `redundant_leaf_key` and count.
- `CLEAVE_TAG_RELEASES=1` (compile time) — tags each CPS release call with its originating `CVar`, readable at the crash. **Caveat: only tensor-typed top-level `PrimOp::Release` sites are tagged**; struct cascades and bufferization-derived releases are not, so a reported tag may be stale from an unrelated earlier call. Verify before trusting it.
- `lldb` from the `clang+llvm-20.1.0` distribution on `PATH` (not `llvm-mlir-22`, which ships no debugger). Batch mode: `lldb -b -s script.txt program.exe`. The single most useful recipe for this bug class:
  ```
  breakpoint set --name cleave_alloc_rc --condition "$rcx == <data_size>"
  run
  bt
  ```
  Win64 puts the first argument in `%rcx`, so this pins down exactly which construction site produced a block of a given reported size. `cdb`/WinDbg cannot launch a child process in this sandbox at all — do not retry it.

---

## 7. Out of scope / known open

- **The `data_size=104` crash — ROOT-CAUSED, and it is *this* bug class, not a separate one.** The earlier guess recorded here (that it fired at construction time and looked "closer to the CSE-merge shape") was wrong on both counts. The offending release is:

  ```
  v2244 = field.b v2239                                  ; read a layer's bias
  Scale::scale<Tensor<f32,1,10>, f32> v2244 v2207 k$30   ; grad * lr
  k$30(v2245)                                            ; v2245 = the call's result
  release v2245                                          ; <- refcount.rs releases it
  Ring::sub<Tensor<f32,1,10>> v2243 v2245 k$31           ; b - lr*grad
  ```

  i.e. the SGD update inside `Optimizer::step`. **`v2245` is a free-standing intermediate tensor returned from a call**: One-Shot Bufferize allocates its buffer (104 bytes = `10 * sizeof(f32) + 64` alignment padding — cleave's own allocator only ever emits exact `sizeof`, confirmed with `CLEAVE_TRACE_ALLOC_TYPES`) *and* emits its `memref.dealloc`, while `refcount.rs` independently releases it because `is_bare_tensor_ty` makes every bare tensor a counted resource. One buffer, refcount 1, two owners.

  Proven, not inferred: tagging the two runtime entry points apart (`cleave_release` = CPS, `cleave_release_void` = bufferization's renamed `free`) shows **323 releases via bufferization and exactly one via CPS** in a run, and the single CPS one lands on the block that then crashes.

  **Three corrections to earlier claims in this file and in session notes**, all from acting on a stale global tag before it was made reliable: the crash is *not* OpenMP-specific (4/6 runs crash with `CLEAVE_NO_OPENMP=1`; the earlier "3/3 clean" was sampling luck), *not* a data race (`OMP_NUM_THREADS=1` crashes too), and *not* located at `field.b` or at `dps_rewrite::Strategy::Overwrite`'s destination. The apparent non-determinism is detection, not occurrence: the double release happens systematically, but only crashes when the pool has not yet recycled the address — otherwise it silently decrements a live block.

  **Diagnostics built for this, kept in the tree**: `CLEAVE_TRACE_SIZE=<bytes>` (a full allocate/retain/release ledger for one size class, usable on the real kernel unlike `CLEAVE_TRACE_RC`), the offending release's own entry point and source position in the fatal message, `CLEAVE_TRACE_ALLOC_TYPES=1` (every LLVM type cleave's own allocator is asked for), and `CLEAVE_NO_DPS` / `CLEAVE_NO_DPS_PASSTHROUGH` for bisecting `dps_rewrite`.
- **The bare-tensor release test asserts on a mechanism, not a property.** `a_bare_tensor_returned_from_a_call_and_consumed_only_as_a_borrowed_argument_is_released` counts CPS `Release` ops rather than checking the allocation is eventually freed. It will trip on any ownership migration regardless of correctness. If Steps 2-3 move a case it covers, reformulate it as a behavioural test rather than weakening it.
- **Arena capacity** is a fixed 256 MB with a hard assert on exhaustion. Routing 153 more allocation sites into it materially raises per-region high-water mark. If `cleave arena exhausted` appears, that is a sizing/growth-policy question (grow-on-demand, or per-region sub-arenas), not a sign the direction is wrong.
- **Thread safety of the arena — resolved by design exclusion, not by making the arena thread-safe.** Corrected from an earlier draft of this section, which recommended a CAS loop; re-reading `doc/hld.md`'s own "Threading" section (line ~224) before implementing changed the answer — see below. `arena_bump` (`cleave-rt/src/lib.rs`) is a plain non-atomic read-modify-write on `ARENA_CURSOR`:
  ```rust
  let cursor = ARENA_CURSOR.load(Relaxed);
  let aligned = (cursor + 15) & !15;
  let new_cursor = aligned + size;
  ARENA_CURSOR.store(new_cursor, Relaxed);
  ```
  Two threads bumping concurrently can be handed the **same** block. `cleave_alloc_rc`/`cleave_release` already guard against exactly this with `POOL_LOCK`, because they are genuinely reachable from OpenMP-outlined `..omp_par.N` regions (confirmed previously against the real kernel's own disassembled object) — the arena has no equivalent guard.

  **`doc/hld.md` states this is a design boundary, not a gap to close**: *"Parallelism inside one tensor op... is opaque to CPS entirely... Nothing in that op's own internal parallel execution ever touches a cleave-level descriptor, refcount, or region — those stay entirely on the orchestrating (CPS) side, which remains single-threaded regardless of how many cores the op it just called used internally."* The region/arena mechanism is meant to stay single-threaded permanently, by construction; genuinely-parallel tensor-op-internal work is meant to route through the (already thread-safe, `POOL_LOCK`-protected) ordinary pool instead. A CAS loop on `arena_bump` would make the race disappear, but it would also legitimize routing opaque-to-CPS, genuinely-concurrent temporaries through a mechanism the design explicitly keeps single-threaded — the wrong fix, even though it would work.

  **The right fix is structural exclusion, and it is cheaper than a CAS loop.** Checked directly against `pipeline.rs`'s own pass ordering: `unify_tensor_allocations` (Step 3's own target) runs right after `--convert-to-llvm` (`pipeline.rs` ~1329), while `omp.parallel`/`omp.wsloop`/`omp.loop_nest` ops still survive as real `omp.*` ops in the module (`--convert-openmp-to-llvm`, run just before, only legalizes their *operand/region types* to the `llvm` dialect — "the ops themselves deliberately survive", that pass's own call-site comment in `pipeline.rs`). The actual outlining into a separate, genuinely-concurrently-invoked function (`__kmpc_fork_call` plus an `..omp_par.N` body) happens only later, during final LLVM-IR translation (`register_all_llvm_translations`), entirely outside the MLIR pass pipeline. **So at the point Step 3's rename runs, a `malloc` lexically inside an `omp.parallel`/`omp.wsloop` region is still textually inside the very same `llvm.func` as its enclosing cleave-level function** — meaning "is this `llvm.func`'s name in `region_local_fns`" (§3's own criterion) is not, by itself, a fine enough check: it would still rename an allocation that is about to become genuinely parallel.

  **The fix**: when walking a candidate `llvm.func` for Step 3's rename, skip any `malloc` call structurally nested inside an `omp.parallel`/`omp.wsloop`/`omp.loop_nest` (or a surviving `scf.parallel`, if `--convert-scf-to-openmp` hasn't run — `options.openmp` false) region — leave those on `cleave_alloc_rc`/`cleave_alloc_auto`'s own pool-backed path, already `POOL_LOCK`-protected and correct today. This is a lexical-nesting check over the already-in-hand LLVM-dialect module (walk up `parent_operation()` from the `malloc` call, the same technique `compensate_refcounts.rs::effective_block` already uses to climb through `scf.if` wrappers — just a different set of op names to stop at), not a dominance analysis.

  **Empirically confirmed to matter, not a hypothetical**: exactly the 45-of-153 target `memref.alloc` sites measured previously as "deeper than the arena boundary" (6 at depth 10, 17 at depth 12, 22 at depth 14 in the pre-`--convert-to-llvm` dump; one spot-checked directly, allocation at line 1551 enclosed by the `scf.parallel` opened at line 1539) are exactly this population. Excluding them is not a smaller/weaker version of Step 3 — it is Step 3 done correctly, per the documented architecture. The remaining 108 shallow sites are Step 3's real, full target.

  `cleave_region_exit`'s unconditional `ARENA_CURSOR.store(handle)` rewind, and `arena_bump`'s own plain (non-atomic) read-modify-write, are therefore both **correct as they stand** and need no change — the exclusion above is what keeps the single-writer invariant true, not a runtime guard. Do not add a CAS loop or per-thread sub-arena as a defensive measure on top; it would suggest a hazard this design deliberately doesn't have.

  If a future workload ever moves cleave to genuine shared-memory multithreading *across* independent CPS computations (`doc/hld.md`'s own explicit "not decided against, just not the design this document currently commits to" caveat), this whole exclusion — and the single-threaded arena itself — would need revisiting then, not before.

---

## 8. Strategy — what the failures actually prove, and what to do instead

**Read this before writing any code.** Five reconciliation attempts are now recorded in §5, plus `suppress_bufferization_owned_releases`'s own shipping shape list. They differ in criterion — shape, allocation provenance, presence of a `memref.dealloc`, retain/release surplus, runtime region depth — and they fail in exactly two ways: they miss a double-owned buffer (crash) or they suppress a release that was the buffer's only owner (leak, 3.6–22 GB every time). None of them is close. That is not five unlucky heuristics; it is one structural fact showing through.

### 8.1 Why post-hoc reconciliation cannot work

`refcount.rs` decides ownership over **CPS values**. One-Shot Bufferize decides ownership over **physical buffers**. The map between them is many-to-many and is computed *after* CPS has already committed its decisions: one CPS value may end up with zero buffers, its own buffer, or a buffer shared with other values; one buffer may back several CPS values. By the time any compensation pass runs, the information needed to recover the intended owner **no longer exists in the IR** — it was never written down, because at the moment each system decided, the other's decisions had not been made.

So no predicate evaluated on a buffer — however clever — can recover it. Every criterion tried is an attempt to *guess* a fact that was destroyed. This is why the failures are symmetric (some miss, some over-apply) rather than converging as the criterion sharpens.

**Corollary, and this is the strategic conclusion: stop reconciling. Eliminate one of the two owners.** There are exactly three ways to do that, and only one of them is cheap.

### 8.2 The three exits, honestly costed

- **(A) MLIR owns every tensor buffer; CPS owns none.** Drop `is_bare_tensor_ty` from `RefcountCtx::is_rc` so `refcount.rs` never releases a bare tensor value. §5 records this as already-failed — but the recorded failure is a *leak*, and we now know why: removing it also removed the release for tensors whose storage cleave itself allocated (struct-field payloads, via `store_native_shape_field`/`dps_rewrite`). Those are a different population, and they are the one place where "cleave owns this" is a real, recorded fact rather than a guess. Separating the two concerns — CPS tracks *struct-field payloads*, never *tensor-typed values* — is the same experiment with the actual bug fixed.

- **(B) CPS owns every tensor buffer; MLIR owns none.** Skip `--ownership-based-buffer-deallocation`. §5: 12.7 GB in 10 s, because `refcount.rs` structurally cannot see most intermediates — they have no CPS node at all (`doc/hld.md` says so explicitly). Making it work means giving CPS a node per intermediate, i.e. re-implementing bufferization upstream. Large, and fights the toolchain.

- **(C) Neither refcounts the hot path** — the region arena this document is named after. Correct in principle and still the right end state, but Step 3 showed it is not *independently* achievable: the arena question ("is this allocation region-local?") is a different question from the ownership question, and getting it wrong yields the same leaks.

**(A) is the one to pursue, and (C) then becomes a pure optimization rather than a correctness mechanism** — which is the right shape for it. Arenas should make dead memory cheap to reclaim, not decide who owns it.

### 8.3 The precondition to check first — cheap, static, and falsifiable

Do not build (A) before answering this, because the whole strategy rests on it:

> **Does every free-standing tensor buffer — every one not reached through a struct field — already have a `memref.dealloc` of its own after `--ownership-based-buffer-deallocation`?**

If **yes**, (A) is safe by construction: dropping the CPS release for those values removes a duplicate owner and nothing else, and the remaining leak surface is exactly the struct-field payloads, which cleave allocates and can therefore release by a rule it actually knows. If **no**, the answer names precisely which values need cleave ownership, and that list — not a shape, not a heuristic — becomes the specification for what `refcount.rs` must keep tracking.

This is a static query over `CLEAVE_DUMP_POST_DEALLOC`'s own output (match each `cleave_alloc_rc`-backed buffer against the set of `memref.dealloc` operands; `compensate_refcounts.rs::memref_behind_ptr` already maps a release's pointer back to its memref, so the machinery exists). It costs one dump and one pass, no rebuild loop, and it either de-risks the entire strategy or redefines it.

#### ANSWERED — measured on the real kernel, and it redefines the criterion

The answer is **no**, but the partition it exposes is clean, syntactic and decidable. Of the **147** `cleave_release` call sites in the post-deallocation dump:

| Population | Count | Owner | Action |
|---|---|---|---|
| Buffer traces to a cleave-built descriptor (`unrealized_conversion_cast` from an `!llvm.struct`) — a struct field's own payload | 103 | cleave | keep the release |
| Buffer is a `memref.alloc` result **with no** `memref.dealloc` — bufferization allocated it and handed ownership over | 20 | cleave | keep the release |
| Buffer is a `memref.alloc` result **that also has** a `memref.dealloc` | **24** | **both — this is the bug** | drop the CPS release |

**So the criterion is not a shape, a provenance bit, or a retain/release tally. It is: "this release's buffer is a `memref.alloc` result that also has its own `memref.dealloc`."** Statically decidable, per buffer, no heuristic.

Two findings fall straight out of this table, and together they explain every failure in §5:

1. **The 24 double-owned buffers are dominated by the `1xN` bias shapes** — 4 each of `memref<1x512xf32>`, `1x256`, `1x128` and `1x10`, i.e. 16 of the 24 — plus a scattering of `32xN`/weight shapes. The `1x10` entries are exactly the `Scale::scale` result behind the `data_size=104` crash (§7).
2. **`suppress_bufferization_owned_releases`'s shipping shape list targets a *different* 24.** It contains `32x512`/`32x256`/`32x128`/`32x10`/`32x784`/`1x784` and no other `1xN` at all — so it suppresses deallocs for buffers that are largely *not* in this table, while leaving every genuinely double-owned bias untouched. It redirects 24 deallocs by coincidence of tuning, not because it found the right 24.

That is the whole bug class, finally located: the pass has always been suppressing the wrong half, which is why new shapes kept crashing and why widening the list leaked.

**Implementation note, and a built-in self-check**: an earlier attempt at exactly this criterion (§5's last row) neutralized **43** releases, not 24, and leaked ~8.9 GB — its in-pass matching (`memref_behind_ptr` + `Value` equality against dealloc operands) disagreed with this static count. Treat **24** as the acceptance test: a correct implementation must neutralize exactly the buffers in row 3 and no others, and `CLEAVE_TRACE_SUPPRESS=1` should report that number. If it reports 43, the matching is wrong — not the criterion.

### 8.4 Validation discipline, non-negotiable

Every one of the five failures above passed its unit tests and was caught only by watching real memory on the real kernel. So:

1. `cargo test -p cleave --release --no-fail-fast` green (re-run a failing `pipeline`/`mlir_lower`/`unify_alloc` target in isolation before believing it — they are JIT/link-flaky under parallel load).
2. `mnist-interop` under §6's memory protocol, **for at least two epochs** — a crash-free short run proves nothing, and every leak above was visible within 10 seconds.
3. Accuracy unchanged (`0.9342`). An ownership bug that stops crashing but corrupts silently shows up here and nowhere else.
4. Hand the wall-clock comparison to the user (10 epochs, ~12–19 s baseline) rather than timing it here.

A change that fixes the crash and leaks is **not** progress toward this goal; §5 now has five entries proving that specific point.

### 8.5 The decisive experiment — both directions leak, which closes the door on compensation entirely

With the criterion from §8.3 implemented structurally (no shapes), the same double-owned population was attacked from **both** sides on the real kernel:

| Which owner was removed | Crash | Memory |
|---|---|---|
| The CPS release (rename `cleave_release` → `cleave_release_noop` for alloc-backed, deallocated buffers) — 43 sites | gone | **~8.9 GB / 40 s** |
| The bufferization `dealloc` (redirect to a decoy for exactly the memrefs a `cleave_release` targets) — 43 sites | gone | **~12.8 GB / 40 s** |

Both remove the crash. **Both leak.** That is the result that matters, and it is not a tuning problem.

If the two operations were genuinely a redundant pair on the same buffer, deleting either one would balance. Deleting either one leaks, so **they are not operating on the same runtime buffer**. One static `memref.alloc` site, one static `memref.dealloc` site and one static `cleave_release` site each stand for *many* runtime allocations inside a loop, and the correspondence between them is not the identity: some dynamic instances are freed by the dealloc, others by the CPS release. Removing either site orphans whichever instances that site was actually responsible for.

**Therefore no static, SSA-level compensation pass can fix this class — in either direction — no matter how precise its criterion.** §8.1 argued this from how the two ownership systems are ordered; this measures it directly, and rules out the remaining possibility that a sharp enough predicate might still work. `suppress_bufferization_owned_releases` is not repairable; it is the wrong kind of thing, and its current shape list only avoids catastrophe by suppressing a small, arbitrary subset.

This removes the last alternative to §8.2's conclusion: **one of the two owners has to stop existing at the source.** Pursue (A) — `refcount.rs` tracks struct-field payloads (where cleave genuinely allocates, and knows it) and nothing else; every free-standing tensor becomes MLIR's, exclusively. Nothing downstream then has anything to reconcile.

### 8.6 CORRECTION — "it leaks" was the wrong rejection criterion, and every verdict above that used it is void

**There are two independent bugs, and this document has been conflating them.**

Measured, with genuinely forced rebuilds (see the warning below — several earlier measurements were not):

| Build | Crash | Epochs before `cleave_alloc_rc: allocation failed` |
|---|---|---|
| **Untouched** (no option A, suppression pass and `dps_rewrite` both intact), `CLEAVE_NO_OPENMP=1` so it survives long enough to observe | — | **4** |
| **Option A** (`refcount.rs` no longer treats bare tensors as refcounted) | gone, 3/3 | **9** |

So the unbounded memory growth is **pre-existing and independent of the double-free**. It is not caused by any of the fixes attempted in §5 — it was already there, and the crash simply killed the process before it could ever be observed. Option A does not merely avoid it: it more than doubles how far the kernel gets.

**Consequently every "crash gone, but leaks N GB → rejected" verdict in §5 is void as stated.** Those fixes were measured against a baseline that leaks at least as badly, and rejected for a property they did not introduce. They may each have been valid; they were judged on the wrong axis. §8.5's "both directions leak, therefore compensation is impossible" conclusion is *not* supported by that evidence either — both directions leak because *everything* leaks — though §8.1's structural argument for why post-hoc reconciliation cannot work stands on its own reasoning and is unaffected.

**Three methodology failures produced this, all worth naming so they are not repeated:**

1. **Env-var-gated compiler flags do not trigger a cargo rebuild of the kernel.** `CLEAVE_NO_DPS`, `CLEAVE_NO_OPENMP`, `CLEAVE_NO_BARE_TENSOR_RC` etc. are read by `cleave.exe` when `build.rs` compiles `kernel.cleave`; cargo's fingerprint does not include them, and `touch kernel.cleave` alone is not always enough. **Always `touch examples/mnist-interop/build.rs` as well, and confirm a `Compiling mnist-interop` line actually appears.** The "disabling `dps_rewrite` gives 5/5 clean runs" result that implicated `Strategy::Overwrite` was measured on a stale binary; re-measured properly it crashes 3/3, so that whole line of investigation was chasing nothing.
2. **Memory was judged from the rising edge of a sawtooth.** Option A climbs to ~19 GB by t=45 s and is back at ~892 MB by t=65 s. A 40-second window shows only the climb and reads as a runaway leak. Watch until the trend is clear, or run to completion.
3. **No baseline control was ever taken.** The untouched kernel crashes in under a second, so its own memory trajectory was never measured — which is exactly why a pre-existing leak went unnoticed for the whole investigation. When the baseline cannot survive, find a configuration that can (here, `CLEAVE_NO_OPENMP=1`) and measure *that* as the control before judging any fix.

**Where this leaves the strategy**: option A is the current best candidate and should be evaluated on its own terms (crash fixed, full test suite, accuracy) rather than against a leak it does not cause. The pre-existing unbounded growth is a **separate open item** and needs its own investigation — start from the fact that the untouched build exhausts memory in 4 epochs with the pool never returning blocks to the OS (`cleave_release` parks every freed block in `FREE_LISTS` forever), which makes size-class fragmentation a prime suspect before anything in this document is implicated.

### 8.7 THE ACTUAL REGRESSION — `8a748f8` is the cause, and `5b2b10c` is known-good

Bisected on the real kernel, with forced rebuilds:

| Commit | Behaviour |
|---|---|
| **`5b2b10c`** "add debug info generation" | **233 MB, perfectly flat. 10 epochs in 28.8 s. `test accuracy: 0.9342`. No crash.** |
| **`8a748f8`** "refcount.rs: fix param-alias composition through Sgd, and a new pass to reconcile cleave/bufferization tensor double-ownership" | Segfaults within 5 s. |
| `346f90e` (backlog doc on top) | Same — crashes 6/6. |

**So the double-free, the memory explosion and the slowdown were all introduced by `8a748f8`** — the commit made earlier in this same investigation *to fix* the double-release bug class. Its parent is clean on every axis this document has been measuring. Everything in §5, §7 and §8.6 was diagnosis performed on top of that regression, which is why no fix ever landed: the base was broken, and "leaks ~9 GB" was a property of the base, not of any candidate fix.

`8a748f8` contains, per `git show --stat 8a748f8`: `refcount.rs` (+1021), `compensate_refcounts.rs` (+583, new — both `compensate_merged_refcounts` and `suppress_bufferization_owned_releases`), `cleave-rt/src/lib.rs` (+271), plus smaller `mlir_lower.rs`/`pipeline.rs`/`cps.rs` changes.

**The `cleave-rt` half of that diff is diagnostics only, and the size-class pool is *not* in it.** The pool landed in `5a29433` and `7f8bccf`, both *before* `5b2b10c` — so the whole point of the memory work ("remove the last 10% still going through the system allocator by amortising allocations through a pool") is already banked and already measured in the known-good base. `8a748f8`'s +271 is `CLEAVE_DEBUG_POOL`, `CLEAVE_COUNT_PARKED_HITS`, `CLEAVE_TRACE_RC`, the parked-block sets, the alloc-serial map and `cleave_release_tagged`. Going back therefore costs nothing but the regression itself.

**What this means for the strategy.** §8.1's structural argument still stands on its own reasoning, but it is no longer urgent: at `5b2b10c` the CPS-vs-bufferization split evidently *works* on this kernel — flat memory, correct accuracy, near-baseline speed. The right move is therefore not to keep patching forward, but to go back:

1. **Restore a known-good base.** Revert `8a748f8` (keeping `346f90e`'s doc). Re-confirm 233 MB / 28.8 s / `0.9342`.
2. **Re-derive what `8a748f8` was actually fixing**, from the clean base — it was written against real double-release crashes, so those were presumably real in *some* configuration. Establish a failing case that reproduces at `5b2b10c` before writing any fix for it.
3. **Re-land its pieces one at a time, each measured against §6's protocol**, instead of as one 2000-line commit. The size-class pool, the `refcount.rs` rework and `compensate_refcounts.rs` are three independent changes; at least one of them costs ~80× memory and the whole benchmark, and a per-commit bisect will say which in a few minutes.

The session's own diagnostics (`CLEAVE_TRACE_SIZE`, the `rc/CPS` vs `void/bufferization` release tagging, `CLEAVE_TRACE_ALLOC_TYPES`) are the right tools for step 2 and should be kept.

## 9. Resolution — what was actually done

Executed, in this order, rather than patching forward:

1. **Every line of the region/arena work was preserved on a branch**, `wip/region-arena` (`5eec57a`): the relaxed `region_analysis.rs` whole-program fixed point with call-site identity (`RegionAnalysis { region_local, safe_sites, sites_by_callee }`), the new `region_specialize.rs` (`f` / `f$region` splitting for genuinely-mixed callees), the arena pool fallback in `cleave_alloc_local`, the `AtomicI64` refcount, and 16 new unit tests. Its commit message records explicitly that all of it was measured on top of the `8a748f8` regression and is therefore **not** re-validated. Nothing is lost; nothing is trusted either.
2. **`main` was reset to `5b2b10c`** and re-verified from a forced rebuild, on every axis at once — not just the one that motivated the reset:

   | Axis | Result |
   |---|---|
   | `cargo test --release --no-fail-fast` (workspace) | **803 pass, 0 fail** |
   | `mnist-interop` resident memory, sampled every 5 s | **233 MB, flat across the whole run** |
   | Wall clock, 10 epochs | **31.0 s** |
   | Accuracy | **`0.9342`** |

3. **Re-landed in small, separately measurable commits** — this document first, then the diagnostics, then any re-derivation of what `8a748f8` was fixing. The rule that came out of §8.6 applies to the re-landing too: a commit that cannot be measured against §6's protocol on its own does not go in.

**A narrowing found while re-landing the diagnostics, and it is the most useful fact to come out of the reset.** `RefcountCtx::is_rc` at `5b2b10c` does **not** include `is_bare_tensor_ty` — the clean base treats bare tensors as refcounted at exactly *one* site (`refcount.rs:1867`, `ctx.is_rc(&ty) || is_bare_tensor_ty(...)`, a single local disjunction), not as a blanket property of `is_rc` itself. **`8a748f8` is what promoted that disjunction into `is_rc`**, making every bare tensor in the whole program CPS-counted. That is precisely the mechanism §7 root-caused the `data_size=104` double-free to: bufferization allocates and deallocates a free-standing intermediate (`Scale::scale`'s result inside `Optimizer::step`), and a blanket-`is_rc` CPS releases it as well. So the double-ownership `compensate_refcounts.rs` was written to "reconcile" was introduced by its own commit, a few hundred lines earlier in the same diff. There is consequently no `CLEAVE_NO_BARE_TENSOR_RC` gate in the re-landed diagnostics: on this base there is nothing for it to switch off.

**The one real open question left.** `8a748f8` was written against genuine double-release crashes, and its `refcount.rs` half claimed a specific bug: param-alias composition through `Sgd` (`return_field_aliases` not composing through a callee that returns one of its own parameters' fields). That fix may well be correct and necessary — but it has never been observed to be necessary *from the clean base*, only from a base that was already broken. **Step 2 of §8.7 stands, and is the next task: produce a case that reproduces at `5b2b10c` before re-applying anything.** If no such case can be produced, the fix was addressing a symptom of the rest of its own commit, and stays out.

**The methodology rule this whole episode earns**, on top of §8.4's four: *a base is only known-good once it has been measured on every axis a change could plausibly damage — tests, memory, speed, accuracy — and re-measured after any reset.* Every wrong conclusion in §5, §7 and §8.6 followed from diagnosing on an unverified base.
