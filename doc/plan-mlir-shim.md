# Plan: cleave talks to MLIR through its own C API only — no `melior`, no `mlir-sys`

## Goal

Drop `melior`, `mlir-sys` and `tblgen` from cleave's dependencies. Rust keeps building the IR (the CPS
to MLIR lowering stays in `mlir_lower.rs`); every call into MLIR goes through one C API that cleave
owns, implemented by its C++ shim. Every rewrite of the IR, every pass pipeline, target configuration
and code generation live in the shim. The shim stays in this repository and compiles with cleave,
against the prebuilt LLVM/MLIR (`cleave-llvm-redist`): it is coupled to the compiler, changes with it,
and one repository with one CI keeps them in step.

The rule this sets: **IR construction in Rust, everything that transforms or compiles IR in C++.**

## Why

- **Redistribution.** Building cleave today goes through three build scripts that each consume the
  LLVM/MLIR toolchain their own way: `mlir-sys` (bindgen, so libclang; `MLIR_SYS_220_PREFIX`; a fork
  for Windows MSVC static linking), `tblgen` (which `melior` runs at build time to generate its
  dialect bindings, a git fork too, `TABLEGEN_220_PREFIX`) and `cleave-mlir-shim` (`cc` against the
  MLIR headers). After this plan there is one: the shim's `build.rs`, compiling the C++ with `cc`
  against the prebuilt LLVM/MLIR and linking it. No bindgen, no libclang, no tblgen, no forks, one
  prefix variable.
- **One extension mechanism instead of three.** IR rewrites currently live in three places: the shim
  (16 functions), Rust over `melior` (`redundant_copy_elim.rs`, `unify_alloc.rs`, and the IR walks in
  `pipeline.rs`), and `melior`'s pass manager. Rust walks over the C API carry its known traps (an op
  removed from its block isn't destroyed; ops are compared by raw pointer).
- **No more working around `melior`.** The shim's `ExecutionEngine` re-implements `melior`'s (whose
  handle is private) to pass a target; `mlir_lower.rs` drops to `mlir_sys` for what `melior` 0.27
  lacks (`mlirOperationCreateParse` for `linalg.matmul`'s indexing maps, attribute arrays, setting an
  attribute) through `from_raw`/`to_raw` (48 sites).

## Where things stand

Measured on the current tree.

- **`mlir_lower.rs`** (8.0k lines): about 160 `append_operation`, 75 generic `OperationBuilder`s,
  ~45 `Type::parse`/`Attribute::parse` (types and attributes are mostly built from text), a few typed
  helpers (`arith::constant`, `func::call`, `llvm::load/store/insert_value/extract_value`,
  `memref::load/store/alloc`, `scf`), value and operation queries (a value's type, its defining op,
  an op's block, moving an op before another), debug-info attributes.
- **`pipeline.rs`**: 22 `PassManager`s (9 of them parsed from a pipeline string), interleaved with 16
  shim calls and Rust IR walks: `mark_mulf_addf_contract`, `insert_stack_scopes_in_loops`,
  `strip_ciface_wrapper_debug_info`, location backfilling, `stamp_target_cpu`, `has_openmp_ops`,
  `has_permuted_transfer` (about 70 walk sites). The matmul schedule is loaded from a file path
  (`matmul_schedule_path`).
- **Rust IR rewrites**: `redundant_copy_elim.rs` (514 lines: `eliminate_self_copies`,
  `lower_dynamic_copies`, `forward_out_param_copies`, `forward_dead_source_copies`) and
  `unify_alloc.rs` (175 lines).
- **Code generation**: `--emit-object` builds a JIT `ExecutionEngine` and dumps its object file. That
  requires every external symbol to resolve at JIT construction, hence `register_unresolved_extern_stubs`
  registering a dummy pointer for each user `extern fn`. The target CPU is both stamped on every
  `llvm.func` (`stamp_target_cpu`) and passed to the engine.
- **The shim** (`cleave-mlir-shim`): 1.6k lines of C++, 16 IR rewrites plus the engine with a target;
  its Rust side declares the externs by hand already, over `mlir-sys`'s handle types.

## Target architecture

One crate, `cleave-mlir` (the current `cleave-mlir-shim`, renamed):

- **C++ side** (`cpp/`): the C API (`cleave_mlir.h`), its implementation, and the passes.
- **Rust side** (`src/`): `extern "C"` declarations written by hand from that header, and a small
  safe layer replacing what `melior` provided: `Context`, `Module`, `Block`, `Region`, `Value`,
  `Operation`, `OperationBuilder`, `Type`, `Attribute`, `Location`. Lifetimes as `melior` has them, so
  `mlir_lower.rs` changes its imports, not its logic. Typed helpers only for the ops `mlir_lower.rs`
  builds often (the list above); everything else through the generic builder.
- **`build.rs`**: compiles `cpp/` with `cc` against the LLVM/MLIR prefix (`scripts/setup-toolchain.ps1`
  fetches it, as today) and emits the link directives for the shim and the LLVM/MLIR libraries it
  uses, which until now `mlir-sys`'s build script emitted.

### The C API

Small and stable, in three groups:

1. **IR construction and queries**, a subset of MLIR's own C API (`mlir-c/IR.h`), re-exported under
   cleave's names: context and dialect loading, module create/print/verify, locations (file/line,
   fused, debug-info attributes), generic operation state (name, operands, results, attributes,
   regions, successors), blocks and regions, values (type, defining op, block argument), operations
   (block, parent, move before/after, erase, set attribute), type and attribute parsing, and parsing
   an operation from text (`linalg.matmul` with indexing maps).
2. **Compilation**: `cleaveRunPipeline(module, pipeline_text) -> error text`. Every cleave rewrite is
   a registered MLIR pass (`cleave-elide-block-copies`, `cleave-lower-spawns{tasks=1}`, ...), so a
   whole stage is one pipeline string, and MLIR's own `--mlir-print-ir-after`/timing work on it. The
   matmul transform schedule is embedded in the shim and loaded from memory, not from a path.
3. **Target and code generation**: one `cleaveTarget` built once from `CodegenOptions` (CPU, with
   `native` resolved in C++; features; optimization level; PIC) and used for both outputs:
   `cleaveEmitObject(module, target, path)` translates to LLVM IR and runs the `TargetMachine`'s own
   object emission, a real AOT path with no JIT and no symbol to resolve, and
   `cleaveJit(module, target)` for `--run` and the tests (symbol registration, lookup, invoke).

## What disappears

- `melior`, `mlir-sys`, `tblgen` and their `[patch.crates-io]` forks; libclang; `MLIR_SYS_220_PREFIX`
  and `TABLEGEN_220_PREFIX`, replaced by one `CLEAVE_LLVM_PREFIX`.
- The shim's parallel `ExecutionEngine`, and the `from_raw`/`to_raw` bridges.
- `stamp_target_cpu`/`stamp_llvm_func_attrs` (the target is the `TargetMachine`'s, set once).
- `register_unresolved_extern_stubs` and the "every symbol must resolve" constraint on
  `--emit-object`.
- The 22 hand-assembled `PassManager`s: `lower_to_llvm` becomes a list of stages, each a pipeline
  string built from the options.
- `redundant_copy_elim.rs`, `unify_alloc.rs` and the IR walks in `pipeline.rs`, rewritten as passes
  in C++, where MLIR's rewriter, dominance and use lists are directly available.
- `matmul_schedule_path` and the runtime file it needs.

## Steps

Each step leaves the tree building and `scripts/test.ps1` green (examples included, since they are
the end-to-end coverage of code generation). One step per commit.

1. **Code generation and target in the shim.** Done (2026-10-09). `cleaveTarget`, `cleaveEmitObject`, `cleaveJit`.
   `emit_object` and `--run` switch to them; `stamp_target_cpu`, the engine re-implementation and the
   extern stubs go. Checks: `--emit-object` objects still link (`digits-interop`, `nanolm`), an
   explicit `--target-cpu x86-64-v2` object has no `zmm` register, a program with its own `extern fn`
   emits without stubs.
2. **The pipeline in the shim.** Done (2026-10-09). Register the 16 existing rewrites as passes; add
   `cleaveRunPipeline`; embed the matmul schedule. `lower_to_llvm` becomes stages of pipeline strings
   (the Rust walks still run between them for now). Check: the lowered MLIR of nanoLM's kernel is
   identical before and after (`--dump-mlir-lowered`, diffed).
3. **The Rust IR rewrites into C++** Done (2026-10-09).: `unify_alloc` first (smallest), then `redundant_copy_elim`,
   then the `pipeline.rs` walks (contract flags, stack scopes, location backfill, debug-info stripping,
   the `has_*` probes). Same identical-output check, plus their existing tests.
4. **The Rust layer.** `cleave-mlir`'s `src/` gets the construction API over group 1 of the C API;
   the shim's Rust side stops depending on `mlir-sys`. `mlir_lower.rs` and the tests switch imports
   module by module. Check: identical `--dump-mlir` output on the examples and nanoLM.
5. **Remove** `melior`, `mlir-sys`, `tblgen` and the patches; the shim's `build.rs` emits the LLVM/MLIR
   link directives itself; `setup-toolchain.ps1` writes `CLEAVE_LLVM_PREFIX` only; `building.md`
   rewritten. Linux then needs only the prebuilt LLVM/MLIR for it and the shim's `build.rs` flags.

Steps 1 to 3 are independent of `melior`'s removal and each pays off on its own (real AOT, readable
pipeline, one place for rewrites); step 4 is the bulk of the mechanical work.

## Risks

- **Identical output is the bar for steps 2 to 4**, not just green tests: a pass moved to C++ that
  matches more or fewer ops than its Rust version changes code silently. Every step diffs the lowered
  MLIR of the examples and nanoLM's kernel before and after.
- **The Rust/C++ boundary.** The C API passes only handles (pointer-sized structs), plain integers
  and `(ptr, len)` strings; no C++ type crosses it.
- **Compile time.** The shim grows from 1.6k lines to several thousand, compiled against MLIR's heavy
  headers. If its share of a cleave build becomes a problem, it moves to its own repository, built
  and published as a prebuilt library alongside LLVM/MLIR; not before, since that splits one change
  across two repositories and two CIs.
- **Object emission without the JIT** must keep what the engine did implicitly: the optimization
  pipeline at the chosen level, debug info (CodeView on Windows), the data layout and triple.
  Compared with `llvm-objdump` on the examples before switching.

## Out of scope

- A cleave dialect (IRDL or TableGen), and the dialects shared with cleave-gfx: explored in
  `doc/backlog.md`; this plan makes them easier (the C++ side and its build exist) but doesn't need
  them.
- Emitting MLIR as text from Rust instead of building it: smaller C API, but a rewrite of
  `mlir_lower.rs`, which queries the IR it builds, and a parse of tens of megabytes per compilation.
