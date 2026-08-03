# Backlog

*Ordered — top to bottom is the order we work through it. Not a wishlist: each entry is a real, confirmed gap (found by testing or by direct inspection), not a guess about what might be missing.*

## 1. Attribute syntax (`#[...]`), with `export`/`extern` as recognized attributes — not separate grammar

A general annotation mechanism on items (`fn`, `struct`, `impl`, ...) — grammar + AST only at first: parses `#[ident]`/`#[ident(args)]`-shaped attributes and carries them as raw, uninterpreted data on the relevant `Item`/`FnDecl` node. Nothing consumes them yet — this exists specifically to close the loop for MLIR lowering later (inlining/linkage-style hints) even though nothing reads them today.

`export`/`extern` turn out not to need their own dedicated keywords/grammar at all — they're just specific, recognized attribute *names*, handled by a later pass that pattern-matches on them, exactly like Rust's own `#[no_mangle]`/`#[export_name]` (as opposed to Rust's `extern "C" fn`, which *is* a dedicated keyword there — cleave doesn't need the distinction):

- **`#[export]`** on a cleave `fn` — generates a thin, C-ABI-compatible wrapper (no name mangling, argument/return types restricted to what a C caller can actually pass) so a compiled cleave function becomes a real symbol Rust/C can link against.
- **`#[mlir(instruction_name)]`** on a *bodyless* algebra-impl method — the concrete mechanism for primitive algebras, more direct than a generic C-ABI `extern`: `impl Float<f32> { #[mlir(mlir_f32_add_instruction)] fn add(x: f32, y: f32) -> f32; }` gives the MLIR lowering pass a direct instruction to emit for any call resolving to this impl, skipping C-ABI/linkage entirely (no calling convention needed for something MLIR emits inline). This is what actually closes the **circular-primitive-impl problem** (`doc/type_inference.md`, "What's still explicitly not done"): `impl Ring<i32> { fn add(x, y) { x + y } }` has no base case today, nothing to bottom out into — an `#[mlir(...)]`-tagged bodyless method gives it one, with everything generic still built in terms of ordinary algebra dispatch on top.

  **Real grammar wrinkle this needs, unlike `export`:** `fn_decl` (used for every `fn`, top-level or impl method) requires a body unconditionally today (`grammar.pest:29`, `FnDecl.body: Block`, non-optional) — the bodyless, semicolon-terminated shape already exists (`fn_sig`, `grammar.pest:48`) but is reserved for an *algebra's own* signature declaration, never usable inside an `impl`. Cleanest fix: merge `fn_decl`/`fn_sig` into one body-optional rule, used everywhere a `fn` appears, and enforce "must have a body" as a *semantic* check instead of a grammatical one — a top-level `fn` or inherent-impl method always requires one; an algebra-impl method may omit it, but only when tagged with an attribute that justifies the omission (`#[mlir(...)]` for now).
  **Extern C-ABI symbols** (calling a *real, separately-compiled* Rust/C function, as opposed to emitting an MLIR instruction directly) may still be a distinct, later need — kept open, not designed yet, since `#[mlir(...)]` alone might cover everything the primitive-algebra case actually needs.

## 2. Real stdlib arithmetic, backed by `#[mlir(...)]`

Every numeric algebra impl in every example so far (`Ring<i32>::add`, `Ring<f32>::add`, `MatMul<f32,f32,f32>::mul`, ...) is a stub returning a hardcoded constant — not because the type checker can't handle a real body, but because there was nothing to call into. Once (1)'s bodyless, `#[mlir(...)]`-tagged methods exist, `stdlib/num/num.cleave`'s own `impl Int<i32>`/`impl Float<f32>` etc. get real primitive bodies this way, instead of `Num`/`Int`/`Float` staying pure markers with zero methods.

## 3. CPS conversion

The actual compilation step after monomorphization — turns a monomorphized, fully-concrete `MonomorphizedProgram` into continuation-passing form. Nothing exists here yet; `monomorphize.rs`'s own output is currently a dead end (rendered via `--dump-monomorphized`, consumed by nothing).

## 4. MLIR lowering

Lowers CPS-form code to MLIR (presumably `arith`/`func`/`scf` dialects to start). The actual "make it run" step.

## 5. Produce a real executable

Wire up an actual compile-to-object/link step (`-c`-equivalent) so a cleave program produces something runnable, not just a dump. Needs (3) and (4) first.

---

*Below this line: pre-existing type-checker gaps, already tracked in `doc/type_inference.md`'s own "What's still explicitly not done" — restated here for one single ordered list. None of these block (1)-(5) above; revisit after.*

## 6. Mutability checking

Nothing checks that a non-`mut` binding is never reassigned — `mutable` is consulted only to decide whether to generalize. Applies equally to plain (`x = v`) and indexed/field (`arr[i] = v`) assignment.

## 7. Monomorphization for inherent-impl methods

`monomorphize.rs` covers top-level `fn`s and algebra-impl methods; a generic inherent method (`impl struct Vec2<T> { fn len(v) {...} }`) has no equivalent worklist treatment yet.

## 8. An inherent method's inferred return type reaching an external caller

Self/mutual recursion *within* one impl block is resolved (`infer_inherent_impl_block`); a *different* function calling an unannotated inherent method still only ever sees a placeholder — nothing propagates a block's own results across impl-block boundaries.

## 9. Self-recursive `let`-bound lambda

`let g = fn(n) { g(n) };` isn't resolved — no name to publish a placeholder under until the lambda is already bound.

## 10. Calling a lambda literal directly

`(fn(a, b) { a + b })(1, 2)` isn't representable — `Call`'s callee is a `Path`, not an arbitrary expression.

## 11. Const-generic algebra parameters

`algebra Foo<const N: i32>` — the const generic is ignored when instantiating the algebra's own declared signature; only `GenericParam::Type` gets a fresh variable there today.

## 12. Scheme satisfiability at generalization time

A scheme's own constraint set is only checked when something actually instantiates it — a mutually-recursive group whose members disagree on shape (`Int t` and `Float t` on the same quantified `t`) generalizes silently if nothing ever calls into it.

## 13. Complex literals

`3+4i` still returns a bare `<complex-not-yet-inferred>` placeholder, never really inferred.

## 14. Bound-satisfiability-aware overlap checking

`check_no_overlapping_impls` is shape-only — doesn't ask whether two impls' respective bounds could really be satisfied by one shared concrete type, just whether the target patterns could structurally coincide.

## 15. Deferred/symbolic constant folding

`[T; N+M]`, where `N`/`M` are const generics still abstract when written — `const_eval.rs` folds pure-literal arithmetic and bare const-generic references today, but a computed expression that only becomes concrete *later* (at instantiation/monomorphization time) needs `Ty` itself to carry a small deferred-expression shape, not just a resolved `Const` leaf or an unresolved `Var`. A real `Ty` extension, bigger than `const_eval.rs` alone.

## 16. `break value` in loops

`for`/`while` can never produce a value (always `Ty::Con("()")`) — no `break value` mechanism. `let mut` plus a post-loop read is the only idiom today.

## 17. Qualified-call syntax

Two algebras legitimately declaring the same operator name (`AmbiguousOperator`) has no way to disambiguate at the call site — an unconditional hard rejection, no qualified-call syntax wired up to bypass it.

## 18. More constant-folding operators

`const_eval::eval_binop` only knows `add`/`mul` today — `sub`/`div`/etc. are one-line additions each, not yet done.
