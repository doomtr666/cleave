# Backlog

*Ordered — top to bottom is the order we work through it. Not a wishlist: each entry is a real, confirmed gap (found by testing or by direct inspection), not a guess about what might be missing.*

## Done

### Attribute syntax (`#[...]`), with `export`/`extern` as recognized attributes

A general annotation mechanism on items (`fn`, `struct`, `impl`, ...) — `#[ident]`/`#[ident(args)]`, carried as raw data on the relevant `Item`/`FnDecl` node. `#[mlir(instruction_name)]` on a bodyless algebra-impl method is the mechanism actually used (below) — the concrete fix for the former circular-primitive-impl problem (`impl Ring<i32> { fn add(x, y) { x + y } }` had no base case to bottom out into). `export`/real C-ABI `extern` symbols stay recognized-but-unconsumed attribute names for now — no backend reads them yet.

### Real stdlib arithmetic, backed by `#[mlir(...)]`

`stdlib/num/num.cleave`: `Ring<T>` (`add`/`sub`/`mul`) and `Ord<T>` (`lt`/`le`/`gt`/`ge`/`eq`/`neq`), each with a real, bodyless, `#[mlir(mlir_<width>_<op>)]`-tagged impl per numeric width (`i8`/`i16`/`i32`/`i64`/`f32`/`f64`) — no more stub bodies returning hardcoded constants.

### CPS conversion — Stages 1-5

Turns a monomorphized, fully-concrete `MonomorphizedProgram` into continuation-passing form (`cleave/src/cps.rs`, `--dump-cps`; 21 tests in `cleave/tests/cps.rs`). See the module's own doc comment for the full design (classical CPS, syntax-directed over the structured AST, primitive-vs-real-call distinction).

- **Stage 1 — straight-line code.** Literals, variables, plain `let`, field access, struct construction, calls — both `#[mlir(...)]` intrinsics as a straight-line `LetPrim` (Appel's PRIMOP) and real/recursive callees as a synthesized continuation + tail `App` (Appel's APP).
- **Stage 2 — `if`/`else`.** Both arms tail-call one synthesized join continuation rather than each inlining `k` separately, which would duplicate "what happens after" combinatorially under nesting.
- **Stage 3 — `while`/`for`.** Loop-carried state via a self-recursive continuation (`for`'s own index; `while` carries nothing extra by itself at this stage).
- **Stage 4 — `let mut`/plain (bare-name) assignment.** Mutation *across* a branch/loop is the real work: `mutated_free_vars`/`mutated_free_vars_expr` (a static, shadowing-aware AST walk, no dominance-frontier/φ-node construction needed) find every enclosing-scope name a branch/loop body might reassign, threaded as extra parameters on the join/loop continuation alongside its own value. Continuations carry `(CVal, &CEnv)` throughout for this reason (state-passing-style CPS).
- **Stage 5 — arrays.** Construction (`ArrayLit`/`ArrayRepeat`, including nested `[[0.0; K]; N]`), and reads/writes at *any* depth (`a[i,j]`/`a[i][j]` — indistinguishable after lowering, and semantically identical in this language: a multi-dim array's own type is always `Array(Array(T,C),R)`). `collect_index_chain` collapses a whole run of nested `Index` nodes into *one* combined, multi-index `Load`/effectful `Store` up front rather than chaining single-index ops — required for a *write* specifically (an intermediate single-index `Load` of "the row", written into separately, would only be correct if `Load` aliased the original storage instead of copying it out — a representation choice never actually made) and reused for reads too, for consistency. `Store` is a real effect, not a functional update: HPC rules out copying a whole array per element write, and a stable array *reference* sidesteps Stage 4's own mutation-threading entirely — identity never changes across a branch/loop, only contents, nothing to carry.

Verified end-to-end on `examples/matmul.cleave`, the project's own canonical target — triple-nested loops, `a.values[i,k]`/`result.values[i,j]`, all converting correctly.

**Two real bugs found and fixed along the way, both by direct testing:**
- **A `for` loop whose own bound names a const generic** (`for i in 0..N`) broke `resolve_synthetic_binop`. Root cause: `infer.rs`'s `ExprKind::For` unifies `start`'s own type variable with `end`'s (here, `N`'s own `const_widths`-tracked one); once monomorphized, the loop index's own declared *width* could come back as `N`'s own resolved *value* (`Ty::Const`) instead of an ordinary `Ty::Con`. Worked around locally in `cps.rs` rather than fixed at the root (see item 6 below): falls back to `i32` — the same default `apply_defaults` itself would have chosen had its own `const_widths` guard not deliberately deferred it, and correct precisely because nothing else ever pins the counter to a more specific width without going through an ordinary `Ty::Con` directly (never hitting this fallback in that case).
- **Nested/chained indexed assignment** (`result.values[i,j] = sum`) initially panicked — an overly conservative single-level-only guard, since a 2D field access desugars to exactly this shape. Fixed by the `collect_index_chain` multi-index collapsing described above, not by relaxing the guard blindly.

**Still explicitly out of scope** (see items 3-4 below): field-mutation assignment (`s.x = v`), and closure conversion (lambdas).

---

## 1. MLIR lowering

Lowers CPS-form code to MLIR (presumably `arith`/`func`/`scf` dialects to start). The actual "make it run" step. Can target whatever Stages 1-5 currently produce — doesn't need items 3/4 below first.

## 2. Produce a real executable

Wire up an actual compile-to-object/link step (`-c`-equivalent) so a cleave program produces something runnable, not just a dump. Needs (1) first.

## 3. Field-mutation assignment (`s.x = v`)

Not attempted at all in `cps.rs` — whether a `struct` is a "light" (rebuilt) or "heavy" (in-place, like an array) value is itself undecided.

## 4. Closure conversion (lambdas)

Deliberately kept out of CPS conversion itself — `hld.md`'s own explicit decision, a separate, later pass: extract each `Lambda`'s body into a fresh top-level `CFunDef`, compute its free variables (captures), represent the lambda value as a code pointer + capture record. `CVal::Label` today only names a *statically known* top-level function — nothing represents an anonymous, capturing function value yet. Items 9/10 below (self-recursive `let`-bound lambda, calling a lambda literal directly) are blocked on this landing first.

---

*Below this line: pre-existing type-checker/stdlib gaps, most already tracked in `doc/type_inference.md`'s own "What's still explicitly not done" — restated here for one single ordered list. None of these block (1)-(4) above; revisit after.*

## 5. Mutability checking

Nothing checks that a non-`mut` binding is never reassigned — `mutable` is consulted only to decide whether to generalize. Applies equally to plain (`x = v`) and indexed/field (`arr[i] = v`) assignment.

## 6. The `Ty::Const`/`Ty::Con` architectural tension, generally

A const generic referenced as an ordinary value (shape slot *and* value at once) keeps causing real, one-off bugs whenever it gets unified with something that isn't itself const-generic-tracked — `check_pending_constraints`'s own `const_widths` bridge, `apply_defaults`'s matching guard, and now CPS conversion's own `for`-loop-bound workaround (see "Done" above) are three separate, narrow patches for the same underlying tension, not a general fix. Worth a real, unified design pass rather than a fourth patch next time this bites.

## 7. Monomorphization for inherent-impl methods

`monomorphize.rs` covers top-level `fn`s and algebra-impl methods; a generic inherent method (`impl struct Vec2<T> { fn len(v) {...} }`) has no equivalent worklist treatment yet.

## 8. An inherent method's inferred return type reaching an external caller

Self/mutual recursion *within* one impl block is resolved (`infer_inherent_impl_block`); a *different* function calling an unannotated inherent method still only ever sees a placeholder — nothing propagates a block's own results across impl-block boundaries.

## 9. Self-recursive `let`-bound lambda

`let g = fn(n) { g(n) };` isn't resolved — no name to publish a placeholder under until the lambda is already bound. Downstream of item 4's own closure-conversion gap — no point resolving this at the type level before there's a way to actually convert/lower the result.

## 10. Calling a lambda literal directly

`(fn(a, b) { a + b })(1, 2)` isn't representable — `Call`'s callee is a `Path`, not an arbitrary expression. Same dependency as item 9.

## 11. Const-generic algebra parameters

`algebra Foo<const N: i32>` — the const generic is ignored when instantiating the algebra's own declared signature; only `GenericParam::Type` gets a fresh variable there today.

## 12. Explicit turbofish on a const generic, for a plain top-level `fn`

`fn rep<const N: i32>(x: i32) -> i32 { N } fn f() -> i32 { rep::<3>(5) }` rejects the call with `` `rep::<...>` expects 0 argument(s), found 1 `` — a real, reproducible bug (a type-generic turbofish, `ident::<i32>(5)`, works fine on the same shape of function). Root cause not yet diagnosed; found while testing CPS `ArrayRepeat` conversion (`sum_n::<3>(1)`-style call, needed to instantiate a const generic without relying on argument-shape inference).

## 13. Scheme satisfiability at generalization time

A scheme's own constraint set is only checked when something actually instantiates it — a mutually-recursive group whose members disagree on shape (`Int t` and `Float t` on the same quantified `t`) generalizes silently if nothing ever calls into it.

## 14. Complex literals

`3+4i` still returns a bare `<complex-not-yet-inferred>` placeholder, never really inferred.

## 15. Bound-satisfiability-aware overlap checking

`check_no_overlapping_impls` is shape-only — doesn't ask whether two impls' respective bounds could really be satisfied by one shared concrete type, just whether the target patterns could structurally coincide.

## 16. Deferred/symbolic constant folding

`[T; N+M]`, where `N`/`M` are const generics still abstract when written — `const_eval.rs` folds pure-literal arithmetic and bare const-generic references today, but a computed expression that only becomes concrete *later* (at instantiation/monomorphization time) needs `Ty` itself to carry a small deferred-expression shape, not just a resolved `Const` leaf or an unresolved `Var`. A real `Ty` extension, bigger than `const_eval.rs` alone.

## 17. `break value` in loops

`for`/`while` can never produce a value (always `Ty::Con("()")`) — no `break value` mechanism. `let mut` plus a post-loop read is the only idiom today (and, since CPS conversion's own Stage 4, actually compiles correctly).

## 18. Qualified-call syntax

Two algebras legitimately declaring the same operator name (`AmbiguousOperator`) has no way to disambiguate at the call site — an unconditional hard rejection, no qualified-call syntax wired up to bypass it.

## 19. More constant-folding operators

`const_eval::eval_binop` only knows `add`/`mul` today — `sub`/`div`/etc. are one-line additions each, not yet done.

## 20. Unary minus (`neg`) not wired up in stdlib

`-x` desugars to a call to `neg` (see `grammar.md`), but no algebra in `stdlib/num/num.cleave` declares it — `Ring<T>` only has `add`/`sub`/`mul`. Any source use of unary minus fails to resolve (`<unresolved-call:neg>`). Found while testing CPS `if`/`else` conversion. One-line-per-width addition once `Ring`'s own impls gain a `#[mlir(mlir_<w>_neg)]`-tagged `neg`, same pattern as the other four ops.
