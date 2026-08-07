# Backlog

*Ordered — top to bottom is the order we work through it. Not a wishlist: each entry is a real, confirmed gap (found by testing or by direct inspection), not a guess about what might be missing.*

## Done

### Real MLIR lowering, end to end (`cleave/src/mlir_lower.rs`)

CPS-form code lowers to real MLIR and JIT-executes (`--dump-mlir`, `--run`) — not just a dump. Covers: a function's own `return`; `if`/`else` as `scf.if`; `while`/`for` as `scf.while`, carrying loop state correctly *per position* (see "real bugs" below — a loop carrying an `i32` counter and an `f32` accumulator at once needs each materialized against its own type, not one shared guess); a real call to another top-level `fn` as `func.call`; an `extern fn` call as `func.call` against a `private` declaration.

**The generic MLIR intrinsic mechanism** — `doc/hld.md`'s own "one generic 'emit this named MLIR op' primitive" thesis, actually built: a reserved `mlir::dialect::op(...)` call (e.g. `mlir::arith::addi(a, b)`), recognized *structurally* by its path's first segment (not algebra-dispatched), lowers via one generic `OperationBuilder`-based builder — zero per-op-name Rust knowledge. Named call-arguments (`predicate: "2 : i64"`) carry a static MLIR attribute's raw text. Replaces the earlier `#[mlir(...)]`-attribute mechanism entirely (deleted, along with `PrimOp::Intrinsic`/`UnitBody::Intrinsic`). `#[mlir_type("...")]` on an algebra `impl` does the same one level up, for *type* lowering (`ty_to_mlir` looks a cleave type name up in a map built from every tagged impl, parses the declared MLIR type text — no per-type-name Rust match left beyond `bool`, a genuine structural special case matching `infer.rs`'s own treatment of it).

**Arrays** — literal/nested-literal (`[[1,2],[3,4]]`)/`[value; N]` construction, and multi-index read/write at any depth (`a[i,j]`, collapsed into one op by `cps.rs`'s own `collect_index_chain`), in **two** representations depending on where the array lives: a standalone array is a self-describing `memref`; an array reached through a struct field (below) is an opaque `!llvm.ptr` into that struct's own storage (a `memref` can't be an `!llvm.struct` field at all — confirmed empirically, MLIR's verifier rejects it, "operand #1 must be primitive LLVM type"), addressed via `llvm.getelementptr` instead.

**Structs** — heap-allocated (a real call to `cleave_alloc`, a new `cleave-rt` function backed by `std::alloc::alloc`, deliberately leaked — no ownership/`drop` story exists yet) reference values, not `!llvm.struct`-typed SSA values built via `undef`+`insertvalue`/`extractvalue`: found necessary by direct testing — a struct returned from one function and read by its caller came back reading garbage once its storage lived in the *constructing* function's own stack frame (`llvm.alloca`, tried first, doesn't survive a return). Field access/mutation (`s.x`, `s.x = v`) go through `llvm.getelementptr` + `llvm.load`/`llvm.store` against this same heap pointer. Works for generic structs (both type and const generics, e.g. `Matrix<T, const R, const C>`) and for array-typed fields (embedded *inline* as `!llvm.array`, not a `memref` — addressed via one combined GEP walking through the struct *and* the array together).

Verified end-to-end via real JIT execution, not just verification: `examples/matmul.cleave` (a generic `MatMul` algebra, triple-nested loops, struct-of-arrays, real generic dispatch — numerically verified against hand-computed expected values, not just "didn't crash") and `examples/complex.cleave` (a generic `Complex<T>` struct dispatched through `Ring`).

**Real bugs found and fixed along the way, all by direct testing, none hypothetical:**
- A `()`-returning top-level function (`fn main() { ... }`, no `->` clause) had no MLIR representation at all — `ty_to_mlir` panicked on the unit type. Fixed: zero MLIR results (not "one result of nothing"), and `CVal::Unit` filtered out of a `return`'s own materialized values rather than reaching `lower_cval` (which doesn't support it).
- `f32`/`f64`/`bool` literal `CVal`s weren't handled in `lower_cval` at all (only `CVal::Int`) — the first program using a bare float/bool literal anywhere but a comparison hit a clean but blocking panic.
- `lower_loop`'s own "before"/"after" region environments started **empty**, not inherited from the enclosing scope — a loop body referencing *any* outer, non-carried variable (an enclosing function's own parameter, chief among them a struct reference) was invisible inside the loop, panicking "unbound CPS variable." Fixed by cloning the outer `env` into each region's own environment before overlaying the loop's own carried params.
- **The deepest one:** `lower_loop` materialized every carried value's own *initial* literal against one shared expected type (the enclosing function's own return type) — silently wrong the moment a loop carries more than one type, or a type different from the function's own result. Masked by every earlier loop test (each only ever carried one type, always equal to the function's own return type) until a loop carrying an `i32` counter *and* an `f32` accumulator at once surfaced it as a native MLIR/LLVM assertion crash, not even a clean panic. Fixed properly, not patched around: `CFunDef` gained a `carried_types: Option<Vec<Ty>>` field, populated in `cps.rs` (`mutated_free_vars`/`mutated_free_vars_expr` now also thread `ctx` and return each mutated name's own concrete type, read off its assignment target's own `ctx.node_types` entry) so `mlir_lower.rs` never has to guess.

### extern fn / cleave-rt / print

`extern fn name(...) -> ...;` declares and calls a foreign C-ABI symbol — no ABI string (`extern "C"`, an earlier draft) needed, the declared signature already says everything else. Backed by a small Rust crate, `cleave-rt`, exposing plain `extern "C" fn` symbols registered with the JIT `ExecutionEngine` by real function pointer (not resolved by dynamic symbol name — sidesteps Windows/MSVC CRT-symbol-visibility questions a raw libc binding would hit). `extern(symbol) fn name(...)` is the override for when several algebra-impl methods share one cleave-level name (`print`) but each needs a distinct real symbol (`print_i32`/`print_i64`/...) — `stdlib/io/io.cleave`'s `Print<T>` is the working end-to-end example, one `impl` per numeric width, `use io;` (not prelude, unlike `num`).

### Real stdlib arithmetic, backed by `mlir::...` calls

`stdlib/num/num.cleave`: `Ring<T>` (`add`/`sub`/`mul`) and `Ord<T>` (`lt`/`le`/`gt`/`ge`/`eq`/`neq`), each with a real body — one direct `mlir::arith::*` call — per numeric width (`i8`/`i16`/`i32`/`i64`/`f32`/`f64`). No stub bodies left. `num` is in the prelude (no explicit `use` needed); `io` is not.

### Field-mutation assignment (`s.x = v`)

A struct is a stable reference, mutated in place — same choice as arrays (below), same reasoning: identity never changes across a branch/loop, only the one field's own storage, nothing to thread as extra carried state. `cps.rs` converts it to a single `PrimOp::FieldStore` (mirrors `PrimOp::Store`'s own "real effect, bound result unit, never read" shape); `mlir_lower.rs` lowers it through the exact same `llvm.getelementptr` field-addressing struct construction/read already use, factored into one shared `store_field` helper.

### CPS conversion — Stages 1-5

Turns a monomorphized, fully-concrete `MonomorphizedProgram` into continuation-passing form (`cleave/src/cps.rs`, `--dump-cps`; tests in `cleave/tests/cps.rs`). See the module's own doc comment for the full design (classical CPS, syntax-directed over the structured AST, primitive-vs-real-call distinction).

- **Stage 1 — straight-line code.** Literals, variables, plain `let`, field access, struct construction, calls — both a reserved `mlir::...` intrinsic as a straight-line `LetPrim` (Appel's PRIMOP) and real/recursive callees as a synthesized continuation + tail `App` (Appel's APP).
- **Stage 2 — `if`/`else`.** Both arms tail-call one synthesized join continuation rather than each inlining `k` separately, which would duplicate "what happens after" combinatorially under nesting.
- **Stage 3 — `while`/`for`.** Loop-carried state via a self-recursive continuation (`for`'s own index; `while` carries nothing extra by itself at this stage).
- **Stage 4 — `let mut`/plain (bare-name) assignment.** Mutation *across* a branch/loop is the real work: `mutated_free_vars`/`mutated_free_vars_expr` (a static, shadowing-aware AST walk, no dominance-frontier/φ-node construction needed) find every enclosing-scope name a branch/loop body might reassign — and, since the MLIR-lowering carried-types bug above, each one's own concrete type too — threaded as extra parameters on the join/loop continuation alongside its own value. Continuations carry `(CVal, &CEnv)` throughout for this reason (state-passing-style CPS).
- **Stage 5 — arrays.** Construction (`ArrayLit`/`ArrayRepeat`, including nested `[[0.0; K]; N]`), and reads/writes at *any* depth (`a[i,j]`/`a[i][j]` — indistinguishable after lowering, and semantically identical in this language: a multi-dim array's own type is always `Array(Array(T,C),R)`). `collect_index_chain` collapses a whole run of nested `Index` nodes into *one* combined, multi-index `Load`/effectful `Store` up front rather than chaining single-index ops — required for a *write* specifically (an intermediate single-index `Load` of "the row", written into separately, would only be correct if `Load` aliased the original storage instead of copying it out — a representation choice never actually made) and reused for reads too, for consistency. `Store` is a real effect, not a functional update: HPC rules out copying a whole array per element write, and a stable array *reference* sidesteps Stage 4's own mutation-threading entirely — identity never changes across a branch/loop, only contents, nothing to carry.

Verified end-to-end on `examples/matmul.cleave`, the project's own canonical target — triple-nested loops, `a.values[i,k]`/`result.values[i,j]`, all converting and (now) actually executing correctly.

**Two real bugs found and fixed along the way, both by direct testing:**
- **A `for` loop whose own bound names a const generic** (`for i in 0..N`) broke `resolve_synthetic_binop`. Root cause: `infer.rs`'s `ExprKind::For` unifies `start`'s own type variable with `end`'s (here, `N`'s own `const_widths`-tracked one); once monomorphized, the loop index's own declared *width* could come back as `N`'s own resolved *value* (`Ty::Const`) instead of an ordinary `Ty::Con`. Worked around locally in `cps.rs` rather than fixed at the root (see the `Ty::Const`/`Ty::Con` item below): falls back to `i32` — the same default `apply_defaults` itself would have chosen had its own `const_widths` guard not deliberately deferred it, and correct precisely because nothing else ever pins the counter to a more specific width without going through an ordinary `Ty::Con` directly (never hitting this fallback in that case).
- **Nested/chained indexed assignment** (`result.values[i,j] = sum`) initially panicked — an overly conservative single-level-only guard, since a 2D field access desugars to exactly this shape. Fixed by the `collect_index_chain` multi-index collapsing described above, not by relaxing the guard blindly.

---

## 1. Produce a real executable

Wire up an actual compile-to-object/link step (`-c`-equivalent) so a cleave program produces a standalone binary. Today's `--run` is JIT-only (`melior::ExecutionEngine`) — proves correctness end to end, but there's no way to hand someone an executable file.

## 2. An `if`/loop whose branches also reassign an outer variable *beyond* its own natural carried state

`mlir_lower.rs::lower_if` only supports a single-parameter join (the `if`'s own value); `lower_loop` only carries what `cps.rs`'s own `mutated_free_vars` already found. Neither path has been extended to a join/loop carrying *more* than that — not yet hit by a real example, but a genuine gap once one does (an `if` whose `then`/`else` both reassign some third, unrelated outer variable, say).

## 3. Closure conversion (lambdas)

Deliberately kept out of CPS conversion itself — `hld.md`'s own explicit decision, a separate, later pass: extract each `Lambda`'s body into a fresh top-level `CFunDef`, compute its free variables (captures), represent the lambda value as a code pointer + capture record. `CVal::Label` today only names a *statically known* top-level function — nothing represents an anonymous, capturing function value yet. Items 7/8 below (self-recursive `let`-bound lambda, calling a lambda literal directly) are blocked on this landing first.

## 4. Dead-code elimination for unused stdlib specializations

Every one of `stdlib/num/num.cleave`'s 36 width specializations (`Ring`/`Ord` × 6 widths) gets emitted into every module regardless of whether a given program actually uses all of them — confirmed harmless for *runtime* performance (LLVM's own JIT optimization inlines the trivial one-op wrappers away, verified via timing), but a real, growing `--dump-mlir` text-size/compile-time cost as the stdlib grows. No dead-code elimination pass exists yet at this level.

## 5. A unit-typed function reachable through a real call, not just the program's own entry point

`lower_top_level_fn`'s own unit-return handling (see "Done" above) is scoped to exactly the program's own entry point (`main`) — a `()`-returning function called *from* another function (via `lower_real_call`) or declared `extern` isn't handled yet; would currently panic clearly in `ty_to_mlir` rather than misbehave, not yet exercised by a real example.

## 6. Nested struct-as-field, unverified

A struct field whose own type is *another* struct — `ty_to_llvm_field_type`'s fallback treats it as a pointer, the same reference representation every other struct value gets — was designed for but never actually exercised end to end by a real test or example this session. Should be verified, not assumed correct.

## 7. Mutability checking

Nothing checks that a non-`mut` binding is never reassigned — `mutable` is consulted only to decide whether to generalize. Applies equally to plain (`x = v`) and indexed/field (`arr[i] = v`, `s.x = v`) assignment.

## 8. The `Ty::Const`/`Ty::Con` architectural tension, generally

A const generic referenced as an ordinary value (shape slot *and* value at once) keeps causing real, one-off bugs whenever it gets unified with something that isn't itself const-generic-tracked — `check_pending_constraints`'s own `const_widths` bridge, `apply_defaults`'s matching guard, and CPS conversion's own `for`-loop-bound workaround (see "Done" above) are three separate, narrow patches for the same underlying tension, not a general fix. Worth a real, unified design pass rather than a fourth patch next time this bites.

## 9. Inherent-impl methods don't reach CPS/MLIR at all

Deeper than a monomorphization gap: `cps.rs::collect_units` has no `ItemKind::InherentImpl` arm at all (only top-level `fn`s and algebra-impl methods are collected as callable units), and `convert_expr` has no `ExprKind::MethodCall` arm either — `v.magnitude_sq()` type-checks fine (`infer.rs`'s own dispatch through `registry.inherent_method` works) but panics at CPS conversion, confirmed by direct testing (`doc/user_guide.md`'s own "Inherent impls" example). `monomorphize.rs` covers top-level `fn`s and algebra-impl methods for *generic* instantiation; a generic inherent method (`impl struct Vec2<T> { fn len(v) {...} }`) has no equivalent worklist treatment either — the same underlying gap, one level up.

## 10. An inherent method's inferred return type reaching an external caller

Self/mutual recursion *within* one impl block is resolved (`infer_inherent_impl_block`); a *different* function calling an unannotated inherent method still only ever sees a placeholder — nothing propagates a block's own results across impl-block boundaries.

## 11. Self-recursive `let`-bound lambda

`let g = fn(n) { g(n) };` isn't resolved — no name to publish a placeholder under until the lambda is already bound. Downstream of item 3's own closure-conversion gap — no point resolving this at the type level before there's a way to actually convert/lower the result.

## 12. Calling a lambda literal directly

`(fn(a, b) { a + b })(1, 2)` isn't representable — `Call`'s callee is a `Path`, not an arbitrary expression. Same dependency as item 11.

## 13. Const-generic algebra parameters

`algebra Foo<const N: i32>` — the const generic is ignored when instantiating the algebra's own declared signature; only `GenericParam::Type` gets a fresh variable there today.

## 14. Explicit turbofish on a const generic, for a plain top-level `fn`

`fn rep<const N: i32>(x: i32) -> i32 { N } fn f() -> i32 { rep::<3>(5) }` rejects the call with `` `rep::<...>` expects 0 argument(s), found 1 `` — a real, reproducible bug (a type-generic turbofish, `ident::<i32>(5)`, works fine on the same shape of function). Root cause not yet diagnosed.

## 15. Scheme satisfiability at generalization time

A scheme's own constraint set is only checked when something actually instantiates it — a mutually-recursive group whose members disagree on shape (`Int t` and `Float t` on the same quantified `t`) generalizes silently if nothing ever calls into it.

## 16. Complex literals

`3+4i` still returns a bare `<complex-not-yet-inferred>` placeholder, never really inferred — unrelated to `examples/complex.cleave`'s own user-defined `Complex<T>` struct, which works fine and needs no built-in literal syntax.

## 17. Bound-satisfiability-aware overlap checking

`check_no_overlapping_impls` is shape-only — doesn't ask whether two impls' respective bounds could really be satisfied by one shared concrete type, just whether the target patterns could structurally coincide.

## 18. Deferred/symbolic constant folding

`[T; N+M]`, where `N`/`M` are const generics still abstract when written — `const_eval.rs` folds pure-literal arithmetic and bare const-generic references today, but a computed expression that only becomes concrete *later* (at instantiation/monomorphization time) needs `Ty` itself to carry a small deferred-expression shape, not just a resolved `Const` leaf or an unresolved `Var`. A real `Ty` extension, bigger than `const_eval.rs` alone.

## 19. `break value` in loops

`for`/`while` can never produce a value (always `Ty::Con("()")`) — no `break value` mechanism. `let mut` plus a post-loop read is the only idiom today (and it actually compiles *and runs* correctly now, end to end).

## 20. Qualified-call syntax

Two algebras legitimately declaring the same operator name (`AmbiguousOperator`) has no way to disambiguate at the call site — an unconditional hard rejection, no qualified-call syntax wired up to bypass it. This is exactly why `examples/matmul.cleave`'s own `MatMul::matmul` is an ordinary named method, not wired to `*` — a second algebra declaring `mul` would make *every* `*` in a program ambiguous, including a genuinely scalar one.

## 21. More constant-folding operators

`const_eval::eval_binop` only knows `add`/`mul` today — `sub`/`div`/etc. are one-line additions each, not yet done.

## 22. Unary minus (`neg`) not wired up in stdlib

`-x` desugars to a call to `neg` (see `grammar.md`), but no algebra in `stdlib/num/num.cleave` declares it — `Ring<T>` only has `add`/`sub`/`mul`. Any source use of unary minus fails to resolve (`<unresolved-call:neg>`). One-line-per-width addition once `Ring`'s own impls gain a real `mlir::arith::negi`/`negf`-calling `neg`, same pattern as the other three ops.
