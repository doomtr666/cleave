# cleave Surface Syntax: Decisions So Far

*Concrete grammar/syntax decisions from early language-design discussion. Complements `hld.md` (architecture) — this file is about surface syntax specifically, not the compiler pipeline. Not a final grammar, a running record to avoid re-deriving these choices.*

## Crates: a directory *is* a compilation unit, `use` is the only mechanism

A real, previously-unaddressed gap: "closing the loop" (the whole standard library — `Int32`, `Float`, `Complex`, `Matrix`, `Ring`, ... — written in cleave itself) can't fit in one file, so a way to organize code across files isn't optional.

Went through two rejected iterations before landing here, both worth remembering:

1. First draft: `mod foo;` (Rust-style compilation-unit discovery, language-owned, no external build tool decides membership — the one thing judged non-negotiable, learned from real, well-documented pain with build DSLs that grow into accidental, poorly-specified languages) plus a separate `module foo { ... }` block for scope (à la C#/C++ namespaces, decoupled from file identity, fixing Rust's own 1:1 file↔module tax).
2. **Rejected: "module" itself is a hollow word** — same critique as `trait` earlier (see `hld.md`'s `signature`/`algebra` naming discussion), and having two separate keywords for what turned out to be one idea was unnecessary ceremony.
3. **Settled: `use linalg;` (or `use linalg::Matrix;`) is the only mechanism, no separate discovery keyword at all.** Resolving it (a compiler-driver concern, not Pest's) searches a path list — the project's own source root, then the shipped standard library — for a directory literally named `linalg`; every `.cleave` file in it, collectively, *is* crate `linalg`. Strict, not just convention: a directory's crate identity isn't redeclarable, and no file outside it can contribute to it. "Crate" (not "module") because it's genuinely one concept now, not two, and because it's a real word rather than an overloaded one.

This keeps the one property that mattered from the Rust-vs-CMake comparison (the language itself, not an external build tool, owns compilation-unit membership) while avoiding two other languages' specific friction: Rust's file↔namespace 1:1 tax (many files can freely make up one crate, no per-file ceremony), and Go's domain-qualified module naming + `replace`-escape-hatch friction (crate names here are short and local, resolved by directory search, never tied to a hosting URL). Genuinely deferred, not solved: global name uniqueness across a public package ecosystem (two independent third-party crates both called `utils`) — not a problem at this project's current scope (one project + the stdlib), matching everything else intentionally left for later.

**Qualified paths (`Ring::add`, `linalg::Matrix<f64, 3, 3>`)** follow from this directly — `path = ident ("::" ident)*`, used everywhere a bare identifier used to be (calls, type references), with the single-segment case still covering the common unqualified name. This is also what a `use` statement actually imports, and what disambiguates when two algebras provide an operation with the same name (see `hld.md`, the earlier "ambiguity when multiple algebras collide" discussion).

**Reconciling this with real distribution (a future Cargo-equivalent, not designed now, just the clean split worth recording):** the compiler only ever does the directory-search resolution above — it never knows about git, registries, or fetching. A separate manifest (analogous to `Cargo.toml`) is the *only* place a short crate name maps to an actual source:

```
linalg = "github.com/cmakekifferur/linalg.git"
```

The (future) package-management tool reads the manifest, resolves/fetches/caches accordingly, and places the result somewhere the compiler's search path already knows to look — the compiler itself never changes. This also closes the "global name uniqueness" question left open above, the same way Cargo already closes it in practice: crate names don't need to be globally unique at all, because the name→source mapping is pinned *per project*, in that project's manifest — two different projects can have a `linalg` pointing at two entirely different repositories with no collision, since resolution is never global.

## Identifiers: Unicode (XID_Start/XID_Continue), not ASCII-only

`π`, `τ`, and other Greek/non-ASCII letters work as identifiers — using Unicode Standard Annex #31's identifier syntax (`XID_Start`/`XID_Continue`, the same standard Rust itself uses for its own Unicode identifier support), not a hand-picked "just Greek letters" range. Strict superset of the plain-ASCII rule it replaces. Fits the target audience directly: mathematicians naturally want to write `π`, `θ`, `Δ` matching how they'd write formulas on paper.

**Adjacent, unrelated gap noticed in passing, not fixed yet:** nothing currently excludes keywords (`let`, `if`, `and`, ...) from also matching the plain `ident` rule elsewhere in the grammar — needs a reserved-word exclusion at some point, not urgent.

## Variadic-looking functions: always `params`-style sugar, never real ABI varargs

cleave-defined functions are never genuinely variadic at the ABI level — a "variadic-looking" function is sugar over a single, ordinary, array-typed parameter (C#'s `params` model): `fn f(args: Array<T>)` stays a fixed-arity function, and `f(1, 2, 3)` desugars at the call site to `f(Array(1, 2, 3))`. Fully type-checked, zero ABI-level variadic calling convention involved — same "reduces to an ordinary, safe call" pattern as operators, structs, and `print`.

Real C-style ABI variadic calling convention (`...`) is reserved exclusively for `extern "C"` declarations, and only for *calling existing external C functions that are genuinely variadic* (e.g. libc's `printf`) — never for cleave's own functions. This turned out to be cheap to support rather than a burden: since cleave already lowers through MLIR/LLVM for all codegen, the hard, platform-specific ABI complexity of variadic calls (register-count conventions, argument promotion rules) is already handled by that backend — LLVM IR has native, clean variadic-call support (`declare i32 @printf(i8*, ...)`, `llvm.va_start`/`va_arg`/`llvm.va_end`), so this isn't a bespoke subsystem cleave needs to build, just a small, bounded piece of grammar (the `...` in an `extern "C"` signature) plus implementing C's default argument-promotion rules at the call site.

**Open, deferred:** writing a generic reduction over a `params` array (e.g. a variadic `max(a, b, c, ...)` implemented as `max(first, max(rest...))`) needs array destructuring/slicing syntax (`arr[0]`, `arr[1..]` or similar) that hasn't been designed at all yet. Not urgent — noted so it isn't lost, not to be worked out now.

## Entry point: `fn main() -> i32`

No special syntax — `main` is an ordinary function, recognized by name convention (like C/Rust), treated as the entry point only when compiling in standalone-executable mode. In library mode (the default ISPC/CUDA-style usage — see `hld.md`, "Interop and deployment model"), there's simply no `main` at all, exactly as a Rust crate compiled as a `cdylib` doesn't have one.

**Returns `i32` directly as an explicit exit code** (C-style), not `Result<(), E>` + a `Termination` trait (Rust's approach). This sidesteps needing sum types (`Result`/`Option`, enums-with-data) to be designed at all for `main` to work — deliberately deferred, not decided here. Command-line arguments, if needed, come from a standard-library function (`env::args()` or equivalent), not a `main` parameter — keeps the entry-point signature minimal.

## The grammar is a funnel: parses more than the pipeline processes yet

`cleave/src/grammar.pest` can be wider than what the rest of the compiler (AST lowering, type inference, the e-graph) actually handles at any given point — a construct parsing successfully doesn't mean it's wired into semantic processing yet. `while`/`for` are a good example of how this actually plays out over time: they parsed (`while_expr`, `for_expr`) long before they had any real type inference behind them, and now they do (see `type_inference.md`, "Control flow gets real inference") — but they still aren't connected to CPS conversion/the e-graph, which is a separate, later stage (see `hld.md`). Parsing, type-checking, and lowering are three genuinely different milestones for any one construct, and this file tracks the first one; don't assume the other two follow automatically just because a construct is documented here.

Also added to the actual grammar (previously only discussed, not yet reflected there):
- **Arrays**: 1D fixed-size type `[T; N]`, literal `[1, 2, 3]`, indexing `a[i]` — matching "Primitives + arrays" below.
- **`implies`**: a fourth logical connective alongside `and`/`or`/`xor`, same plain-strict-function treatment, but at its own, lower precedence level — standard propositional-logic convention (`a and b implies c or d` reads as `(a and b) implies (c or d)`).

## Bindings: `let` / `let mut`, plain `=`

Kept explicit `let`/`let mut` keywords (considered and rejected two alternatives: `<-` for binding, and inferring "new binding vs. reassignment" purely from whether the name already exists in scope).

- **`=` for binding, not `<-`.** `<-` was floated to visually distinguish binding from the mathematical `==` used in axioms, but `=`/`==` already coexist without confusion in essentially every mainstream language (C, Rust, Python, Julia, MATLAB) — including ones the target audience already uses daily. No strong enough reason to depart from the near-universal convention.
- **Inferring `let` vs. `mut` from "is this name already bound" was rejected.** It would conflate **shadowing** (`let x = parse(s); let x = x.trim();` — a fresh, unambiguous rebinding, common and legitimate, semantically free exactly like any other `let`) with **mutation** (needs the def-use analysis below) — two things that must stay distinguishable. It would also require scanning the whole enclosing scope (for both compiler and human reader) to classify a single binding site, rather than that being locally visible.
- **`let a = ...`** (plain, immutable): needs no def-use analysis at all. Lexical scoping already resolves which definition a use refers to, unambiguously and structurally. Copy propagation here is free (see `hld.md`, "Constant and copy propagation").
- **`let mut a = ...; ...; a = ...;`** (reassignment under branches/loops): the real ambiguity case. Resolved via a local def-use check attached as a *condition* on the copy-propagation rewrite rule, not full SSA/dominance-frontier construction (see `hld.md` for why the latter isn't needed here). `let mut` was deliberately kept as a first-class, ergonomic construct — loops with in-place accumulation are the daily bread of HPC code; forcing tail-recursive accumulator style instead (even with syntactic sugar) was considered and rejected as a real ergonomic regression for the target audience, not a viable simplification.
- **`+=` and friends**: not decided now, can be added later as pure sugar for `a = a + b` without engaging anything else in this document.

## Blocks are expressions: a trailing `;` decides whether the last expression is the block's value

**A real gap, found only by someone actually hitting it, not by inspection: this rule existed in the grammar since early on but was never written down anywhere.** Recorded here after the confusion it caused was reported directly ("la syntaxe `;` ou pas `;` est parfaitement imbitable").

`block = { "{" ~ stmt* ~ expr? ~ "}" }`, and `expr_stmt = { expr ~ ";" }` (`grammar.pest`) — the same convention Rust and OCaml both use, not invented here: a block is a sequence of `;`-terminated statements, optionally followed by *one more* expression with **no** trailing `;`. That final, semicolon-less expression (the **tail**) is the block's own value. If every expression in the block is `;`-terminated — including, easy to miss, the *last* one — there is no tail, and the block's value is `()` (unit), regardless of how meaningful the discarded expression's own value looked.

```
fn f(x) { if x > 2 { g(x) } else { x } }   // tail = the if-expression; f returns whatever it returns
fn f(x) { if x > 2 { g(x) } else { x }; }  // trailing `;` — no tail; f returns () no matter what g/x compute
```

Both parse. Both type-check (assuming `g`/the comparison resolve). They are not the same function — the second one's body value is always `()`, and everything the `if` computed is thrown away the instant that block ends. This is exactly what happened in practice with a self-recursive `fibonacci`: an extra trailing `;` after the `if`-expression made `fibonacci`'s own inferred type `(t) -> ()` instead of `(t) -> t`, which then failed elsewhere with `no impl Num<()>` — a real, correct type error, but one that doesn't *read* like "you have a stray semicolon" at all unless you already know this rule. `Infer`'s own diagnostics now hint at this directly whenever `()` shows up somewhere it doesn't belong (see `infer.rs`'s `TypeErrorKind` `Display` impl) — but the rule itself belongs here, not only inferred from an error message after the fact.

Same rule applies to `if`/`while`/loop bodies (each arm is its own `block`) and to lambda bodies — there's no special case anywhere; a block is a block.

## Operators: sugar for named algebra functions, minimal set, fixed precedence

**Core principle:** an operator is not a distinct language concept. `a + b` desugars (as early as possible, ideally at parse time) directly to `add(a, b)` — an ordinary named function call, dispatched by the same algebra mechanism as any other call. There is no separate "operator overloading" system (unlike C++'s `operator+`, with its own grammar and resolution rules interacting with ADL/implicit conversions). "Overloading `+`" for a new type is not a distinct feature — it's just supplying `add` in that type's algebra `impl`, which you'd do anyway.

**Desugars to the algebra-qualified name, `Ring::add(a, b)`, never a bare global `add(a, b)`.** This is exactly how Rust avoids operator/free-function name collisions — `a + b` desugars to `<T as Add>::add(a, b)`, never to a plain-name lookup. A user's own standalone `fn add(a, b) { a + b }` (an ordinary, separately-namespaced function, inferred to `fn add<T: Ring>(a: T, b: T) -> T`) never collides with `Ring::add` itself — they live in different namespaces, and calling `v1 + v2` always resolves via the algebra qualification, dispatched to the concrete `impl Ring<Vec2>` once types are monomorphized.

**Guard against circular algebra implementations.** Writing `impl Ring<Vec2> { fn add(a, b) { a + b } }` with `a, b: Vec2` directly is a genuine bug, not just a style question: `a + b` there desugars to `Ring::add` for `Vec2` — the same implementation being written — infinite circular recursion, no base case. A correct implementation bottoms out at the field level (`Vec2(a.x + b.x, a.y + b.y)`, where `a.x`/`b.x`: `f64` dispatch to `Ring::add` for `f64`, a different, more primitive `impl`). The compiler should detect a circular algebra implementation with no base case and reject it — the same class of guard already established for unbounded recursive-continuation inlining.

**The set of operator symbols is fixed and closed — not user-extensible.** What's open is which types support them (via algebra `impl`s), never new symbols/fixity/precedence (avoiding the complexity Haskell/Scala's user-definable operator precedence is known for).

**Minimal operator table** (3 precedence levels, standard mathematical convention, left-associative except noted):

1. unary `-`
2. `*`, `/`
3. `+`, `-`
4. comparisons (lowest): `<`, `<=`, `>`, `>=`, `==`, `!=`

**Explicitly *not* given operator syntax — named functions instead:**
- Bitwise ops (`bitand`, `bitor`, `bitxor`, `bitnot`, `shl`, `shr`) — rare in numerical/HPC code, no reason to spend a precedence level on them. Also sidesteps C's infamous `&`/`|` vs. `==` precedence trap entirely, since there's no bitwise *symbol* to collide with anything.
- Exponentiation (`pow(a, b)`) — sidesteps the classic `-2**2` prefix/associativity ambiguity trap.
- Modulo — named function, not an operator; **naming needs care**: `mod` and `rem` differ in behavior on negative operands (mathematical modulo vs. truncated-division remainder) and must be named/distinguished explicitly, not collapsed into one arbitrarily-chosen name.
- `++`/`--` (prefix or postfix), rejected entirely, not deferred. Not a clean case of "sugar for a pure function call" at all — it's a mutation-with-embedded-read baked into an expression, position-dependent in meaning, and a well-documented source of unsequenced-behavior bugs (`a[i++] = a[++i]`-style). Saves a few characters against `x = x + 1` at a real, recurring cognitive cost. `x = x + 1` (or a future `+=`) covers the same need with no ambiguity.

**Logical `and` / `or` / `xor`: plain, strict Bool-algebra functions, not special forms.** No short-circuit evaluation, no desugaring to `if` (an earlier, more elaborate proposal — desugar to `if` to get lazy evaluation for free from CPS — was considered and superseded by this simpler one). Both operands are always evaluated, exactly like `add`/`bitand`/any other algebra call — zero special-casing anywhere in the compiler. This is a genuine, deliberate divergence from `&&`/`||` in most mainstream languages, and must be documented clearly as such. If short-circuit/lazy evaluation is genuinely needed (e.g. guarding against evaluating an operand that would error), write the `if` explicitly — the laziness becomes visible in the code rather than hidden behind an innocuous-looking `and`. Judged to cost little in practice for numerically-oriented code, where `and`/`or` mostly combine side-effect-free comparisons.

Written in words (`and`/`or`, not `&&`/`||`), matching Python — familiar to the target audience, and avoids any visual confusion with bitwise symbols (moot anyway since bitwise ops aren't operators here at all).

## Structs implement algebras, à la Rust

Generalizes the pattern already established for `Complex` (structural, not opaque — see `hld.md`) to arbitrary user-defined product types.

**Superseded from an earlier sketch, recorded here rather than silently overwritten — implementation actually landed on something different, once construction syntax became a real, concrete question rather than a sketch:**

```
struct Vec2 { x: f64, y: f64 }

impl Ring<Vec2> {
    fn add(a: Vec2, b: Vec2) -> Vec2 { Vec2(x: a.x + b.x, y: a.y + b.y) }
    // ...
}
```

- **Construction is named-argument call syntax: `Vec2(x: 1.0, y: 2.0)`, not positional `Vec2(1.0, 2.0)`.** The original sketch didn't settle this; once it became a real question, `Vec2 { x: 1.0, y: 2.0 }` (Rust's own syntax) was considered and rejected — it collides with `if`/`while`/`for`'s own condition-then-block shape (`if x { y }` — is `x { y }` a struct literal, or is `{ y }` the `if`'s block?), the exact ambiguity Rust itself has to special-case away (disallowing a bare struct literal in condition position). Reusing `(...)` sidesteps it entirely, at the cost of a narrow, separate ambiguity between a zero-arg call and a zero-*field* struct construction (`Empty()`) — resolved by always preferring the call reading, with a single fallback case in `infer_call` recognizing a zero-arg call naming a known, zero-field struct (see `grammar.pest`'s `primary` comment, `cleave/src/infer.rs`). Every field must be named — no positional construction, deliberately, so a construction site names what it means rather than relying on argument order matching a declaration that might be far away.
- **`v.x` is a direct field access (`ExprKind::FieldAccess`), not sugar for an auto-generated accessor function `x(v)`.** The original sketch's "auto-generated field-accessor functions, `v.x` desugars to `x(v)`" wasn't built — field access resolves directly against the struct's own declared fields during type inference (`Infer`'s `FieldAccess` handling), without going through the algebra-dispatch machinery operators use. Revisit if a real need for `x` as a free-standing, algebra-dispatchable function ever comes up (e.g. `map(x, vectors)`) — not needed for the base case.
- **`struct` (data/layout) and `impl algebra for Struct` (behavior) stay separate**, mirroring Rust's own struct/trait-impl split — nothing new needed beyond the already-established algebra mechanism.
- **Generic structs (`struct Pair<T> { a: T, b: T }`) construct and infer correctly now** — `Ty::App(name, type_args)` (added specifically to close this gap) represents "a type applied to generic arguments", so `Pair(a: 1.0, b: 2.0)` infers `Pair<f64>` by unifying both fields' values against the struct's own shared `T`, with no annotation needed. See `type_inference.md`, "Generic structs: `Ty::App` and const-generics", for the mechanism and for the earlier state this note used to describe (a real, previously scoped-out gap, not always the case).

## Generics: explicit `::<...>` (turbofish), never bare `<...>`

**A real grammar ambiguity, not just a design preference.** The natural-looking `f<T>(x)` (bare angle brackets for an explicit type argument at a call site) is genuinely ambiguous in this grammar: `f < T > (x)` also parses as a chained comparison (`(f < T) > (x)`), and PEG's ordered-choice, no-backtracking-across-alternatives nature means there's no clean way to try both readings and prefer whichever "looks more like generics" — the grammar has to commit to one interpretation structurally, not heuristically. Rust hits the same ambiguity and resolves it with the "turbofish" (`f::<T>(x)`) for exactly this reason; cleave adopts the identical `::<...>` syntax, for the identical reason, rather than rediscovering a worse workaround.

```
fn id<T>(x: T) -> T { x }
let a = id::<f64>(1.0);         // pins T explicitly, even though it's inferable here
let v = Vec::<f64, 3>(data: [1.0, 2.0, 3.0]);  // struct construction, mixing a type and a const-generic argument
```

Turbofish is an **override**, not a requirement — omit it whenever inference already pins the generic down unambiguously (the overwhelming common case); reach for it when a literal's own shape would otherwise default somewhere unwanted (`id::<f64>(1)`, forcing the bare integer literal `1` to be treated as `f64` rather than defaulting to `i32`), or when constructing a generic struct with no field value that would otherwise pin every parameter (an empty array, a zero-field generic marker type).

## Function types: `(T) -> U`, for higher-order parameters

A parameter (or any other type-annotation position) can be typed as a function itself:

```
fn apply(f: (i32) -> i32, x: i32) -> i32 { f(x) }
fn compose(f: (f64) -> f64, g: (f64) -> f64, x: f64) -> f64 { f(g(x)) }
```

`fn_type = "(" (type_ ("," type_)*)? ")" "->" type_` — added as `type_`'s first alternative, ahead of the ordinary `path`-based case, so `(` unambiguously starts a function type wherever a type is expected (nothing else in type position can start with `(`). Unlike an ordinary `fn_decl`'s own return type, the `->` here is **mandatory**, not optional: a ordinary function declaration can omit it and let the body's own inferred result fill it in, but a bare type annotation has no body anywhere nearby to infer a return type *from* — leaving it out would just be silently wrong (defaulting to `()`) rather than an honest gap. This is what actually makes a function a legitimate value to pass around — `apply`'s own `f` parameter can be handed any lambda or named function whose shape matches, checked structurally like any other type.

## Inherent impls: methods without an algebra, `impl struct Vec2 { ... }`

Not every method belongs to an algebra — sometimes a struct just needs an ordinary method, with no operator-dispatch/multi-implementation story attached (`v.len()`, not "some algebra `Norm` that any number of unrelated types could also implement"). An **inherent impl** covers exactly this:

```
struct Vec2 { x: f64, y: f64 }

impl struct Vec2 {
    fn len_sq(v) { v.x * v.x + v.y * v.y }
}

impl<T> struct Boxed<T> {
    fn get(b) -> T { b.value }
}
```

**The literal `struct` keyword right after `impl`/`impl<...>` is not decoration — it resolves a genuine grammar ambiguity, found by direct testing, not just an ordering concern.** `impl<T> Boxed<T> { ... }` (no `struct`) parses *identically* two different ways: as an algebra impl (`algebra` = `Boxed`, target = the bare type `T`) or as an inherent impl on the generic struct `Boxed<T>` — both fully match the same character sequence. Since `algebra_impl` is tried first in the grammar's ordered choice, it always wins, silently misreading "Boxed" as an algebra name (`` `get` is not declared by `algebra Boxed` `` — the actual error this produced before the fix). The mandatory `struct` keyword breaks the tie the same way Rust's own `impl Trait for Type` vs. `impl Type` split does (`for` as the disambiguator there, `struct` here) — reusing an existing keyword rather than inventing a new one, and deliberately scoped to the inherent-impl side only, so the pervasively-used ordinary algebra-impl syntax (`impl Ring<i32> { ... }`) needed no changes at all.

**No implicit `self` — the receiver is an ordinary, explicit first parameter.** `v.len_sq()` calls `len_sq` with `v` filling its first positional parameter, exactly the same "no implicit anything" posture the rest of the language holds to. An unannotated first parameter defaults to the impl's own target type (`Vec2` for `len_sq`'s `v`, `Boxed<T>` for `get`'s `b`) — the same "fall back to what the enclosing context already establishes" treatment an algebra impl's own unannotated parameters already get from that algebra's declared signature; an inherent impl has no separate signature to fall back to, so the target itself is the closest equivalent. Any *other* parameter, and an explicitly-annotated first one, are checked/resolved normally.

## Heterogeneous algebras: more than one type parameter

An algebra isn't limited to a single generic type — `algebra MatMul<A, B, C>` genuinely relates three independent types, not one type used three times:

```
algebra MatMul<A, B, C> {
    fn mul(a: A, b: B) -> C;
}

struct Matrix<T : Float, const R : i32, const C : i32> { values : [T; R, C] }

impl<T: Float, const N: i32, const M: i32, const K: i32>
    MatMul<Matrix<T,N,M>, Matrix<T,M,K>, Matrix<T,N,K>> {
    fn mul(a, b) { /* ... */ }
}
```

`algebra_impl = "impl" generic_params? ident "<" type_ ("," type_)* ">" "{" fn_decl* "}"` — the comma-separated repetition after the first target is what actually enables this; a single-target algebra impl (`impl Ring<i32>`, the overwhelming common case) is just this shape with zero repetitions, fully backward compatible. This is the syntax `a * b` for two differently-shaped matrices resolves through — see `type_inference.md`, "Heterogeneous algebra dispatch", for how the compiler resolves the shared/output-only type parameters (`M` shared between the two operands, `N`/`K` appearing only in the *result*) this genuinely needs.

## Algebra bounds: `algebra Int<T> : Num { ... }`

An algebra can declare that it requires another one, the same relationship Rust expresses with a supertrait (`trait Int: Num`):

```
algebra Num<T> {}
algebra Int<T> : Num {}
algebra Float<T> : Num {}

impl Int<i32> {}   // no separate `impl Num<i32>` needed — `Int<i32>` counts as `Num<i32>` too
```

`algebra_decl = "algebra" ident generic_params? (":" bound_list)? "{" algebra_item* "}"` — this was already part of the grammar (`AlgebraDecl::bounds`) well before it meant anything; for a long stretch it parsed and was carried across file merges with nothing ever consulting it, which is why `stdlib/num/num.cleave` used to declare `Int`/`Float` as flat, independent markers duplicating `Num`'s own impls by hand. It's real now — see `type_inference.md`, "Algebra-bound inheritance", for the actual checking mechanism (and why it lives in `Infer`, not `Registry`).

## Primitives + arrays as the minimum type set; tensors/matrices are algebras, not arrays

**Minimum primitive set:** Rust's primitive types (`i8`..`i128`, `u8`..`u128`, `f32`/`f64`, `bool`, `char`) as the concrete starting point, plus arrays. The earlier-discussed `N`/`Z`/`R`/`C` mathematician-facing surface vocabulary and WASM-style signed/unsigned-as-operation refinement (see `hld.md`) can layer on top later as sugar — not required to start wiring the core.

**Arrays fit directly into the already-established light/heavy distinction** (see `hld.md`, "Light vs. heavy storage") — a small fixed-size array of light elements can itself be light (register-resident); a large or dynamically-sized array is heavy. No new concept needed.

**Multi-dimensional arrays: nested types already work, no grammar change needed for the semantics — but Fortran-style sugar is worth it for ergonomics.** `[[f64; 4]; 3]` (a fixed-size array of arrays) parses today with zero grammar changes (`array_type`'s element position is `type_`, itself recursively `array_type`-capable) and gives exactly the same contiguous, row-major memory layout a native 2D array would. But nested-bracket syntax and nested indexing (`a[i][j]`) are a real ergonomic step down from Fortran's `A(3,4)` / `A(i,j)` — and array ergonomics are arguably the one thing that keeps Fortran, otherwise a rough language by modern standards, genuinely loved by the scientific community. Worth the small grammar cost: `[f64; 3, 4]` (comma-separated dimensions) and `a[i, j]` (comma-separated indices) are added as pure sugar over the same nested form — no new semantics, no new memory layout, just surface convenience matching what scientists already expect.

**Array repeat literal: `[value; N]`, sugar expanded at *lowering* time, not a runtime operation.** The natural-looking alternative — a stdlib function, `repeat(value, N)` — isn't actually buildable in cleave today: there's no way to construct or mutate an array element-by-element in a loop yet (no index-assignment, `assign_stmt` only reassigns a whole binding), so a real `repeat` would need much larger array-mutation primitives to exist first, unrelated to just wanting a repeated value. `[value; N]` sidesteps that entirely by desugaring, in `lower.rs`, directly into the ordinary `N`-element form (`[value, value, ..., value]`) — exactly the same "expand at lowering time, add zero inference-side machinery" move `[f64; 3, 4]` above already makes for multi-dimensional sugar. `N` is restricted to a literal integer specifically (`array_repeat = expr ~ ";" ~ numeric_lit`, a distinct grammar rule from the ordinary comma-separated `array_list`, so lowering can tell which one it's looking at) — lowering is a purely syntactic pass with no unification or const-generic resolution available to it, so it can only ever expand a count it can read directly off the token stream; a *variable* repeat count is a real, different, harder problem (needs the array-mutation primitives above) and isn't attempted here.

**Tensor/Matrix/Vector must be their own structural algebras, not "arrays with a specific MLIR lowering."** This is a direct consequence of the founding "don't lower prematurely" thesis (see `hld.md`, "Core philosophy") — the whole motivating example for that thesis (`matmul(matmul(a,b),c)` vs. `matmul(a,matmul(b,c))` reassociation for FLOP savings) requires the e-graph to see `MatMul` as a named operation with real algebraic axioms (associativity, distributivity), not as ordinary array-indexing loops indistinguishable from any other loop. Same relationship as "primitive types are algebras too": Matrix's *eventual* MLIR lowering will of course use array-backed memory, but its language-level identity and the e-graph's reasoning about it stay at the structural/named-operation level — that's a lowering detail, not the type's identity.

**Acceptance test for the struct+algebra mechanism, now met:** `Matrix<T, Rows, Cols>` (real generics — element type *and* const-generic dimensions) is defined *entirely* via ordinary `struct`+`impl algebra` today, no compiler-special-casing — `struct Matrix<T : Float, const R : i32, const C : i32> { values : [T; R, C] }` plus a heterogeneous `MatMul` impl (see "Heterogeneous algebras" above) validates "closing the loop" (see `hld.md`) for compound generic types, not just scalars. See `type_inference.md` for the const-generic (`Ty::Const`) and generic-struct (`Ty::App`) machinery this relies on.

**Backing representation: a Fortran-style array descriptor, not a Rust-style contiguous slice.** `&[T]` (pointer + length) assumes stride-1 contiguous memory — too narrow for real HPC needs (sub-matrix views, transposed views, strided access for FFTs, halo regions). A descriptor generalizes this with an explicit stride and bound per dimension — `struct ArrayDescriptor<T, const RANK: usize> { ptr: *T, strides: [isize; RANK], shape: [usize; RANK] }` — the standard, well-precedented representation (Fortran's own array descriptors, and why numpy views work the way they do).

- **Transpose/reshape become metadata-only, zero-copy operations**: `Transpose(M)` just swaps stride/shape entries, same base pointer, no data movement — keeps `Transpose`/`MatMul` cheap to explore structurally rather than forcing a copy at every rearrangement.
- **The descriptor itself is light, what it describes is heavy** — another instance of the already-established light/heavy split (see `hld.md`): a small, fixed-size descriptor (statically known rank) can be register-resident/passed by value even though the memory it points to is large and heap-allocated. The same pointer-is-light/payload-is-heavy split as `&[T]` in Rust, generalized to arbitrary strides.
- **The descriptor is itself just an ordinary `struct`**, defined in the standard library with the same struct+algebra mechanism as everything else — not a compiler-magic type. `Matrix`/`Tensor` are algebras whose concrete `impl` uses this descriptor as backing storage.

## Complex needs one piece of dedicated literal grammar — nothing else

Its type and operations stay ordinary `struct`+`algebra` (no compiler-special-casing) — but its **literal syntax** is a genuine, narrow exception, same category as numeric literals themselves or the `:type` suffix needing dedicated lexer rules.

**Why `i` can't just be an ordinary named constant like `π`.** `π` is one fixed real value — an ordinary constant suffices. `i` is tied to the precision of the underlying float (`Complex<f32>`'s `i` and `Complex<f64>`'s `i` are different concrete values, not "the same constant") — a named constant would have to commit to one precision or need its own polymorphism story. A *literal* sidesteps this for free by inheriting the exact same default/override precision rules already established for ordinary numeric literals (`2i` defaults like `2.0`; `2i:f64` forces `f64`, same mechanism as `1.25:f64`).

**The grammar rule needed:** a number immediately followed by `i`, no whitespace (`<digits>[.<digits>][e<exponent>]i`) — a dedicated "imaginary literal" lexeme, since numbers can't otherwise be followed by a letter (not a valid identifier — can't start with a digit — and not an ordinary numeric literal either). No juxtaposition-as-multiplication anywhere else in the grammar; this is the one dedicated exception.

**Canonical, always-available lowered form:** every complex literal desugars to the ordinary struct constructor, `Complex(re, im)` — `3+4i` is exactly `add(Complex(3.0, 0.0), Complex(0.0, 4.0))`, i.e. `Complex(3.0, 4.0)`, always writable directly and equivalently by hand, never a hidden or special form underneath the sugar.

## Type conversion

**No implicit conversion anywhere** — already settled by the earlier strict-typing decision (no int/float mixing), matching Rust exactly (`as`/`.into()` always explicit, even for value-preserving widening like `i32` → `i64`). Two genuinely different operations, kept sharply distinct:

- **Value conversion** (`i32` → `f64`, an actual recomputation) is an algebra like any other: `algebra From<Source> { fn from(s: Source) -> Self; }`. Primitive conversions are intrinsics (`#[mlir_lower = "arith.sitofp"]`, same pattern as `add`); any user type can supply its own `impl From<X> for Y` — same extensibility mechanism as everything else, nothing new.
- **Bit reinterpretation** (`i32` ↔ `u32` at equal width, no value computation — the "same storage, different algebra" idea from the WASM-style signed/unsigned discussion) stays a separate, deliberately loud operation, e.g. `bitcast::<u32>(x)` — never conflated with `From`, since `transmute`-style bit reinterpretation and value conversion give different, easily-confused results (`-1i32` bitcast to `u32` is `4294967295`; there's no single obvious "value conversion" of `-1` to an unsigned type).

**Resolved: bare-literal friction (`a = a + 1;` where `a: f64`) needs no exception at all.** Not a case of implicit conversion — the literal `1` was never committed to `i32` in the first place, it's inferred as `f64` directly from the expression context, the same literal-type-inference mechanism already established for defaults/`:type` overrides (see `Complex` above), just also driven by surrounding context. This is exactly the line Rust itself already draws: strictly no implicit conversion between two already-typed variables, but ordinary contextual inference for untyped literals. Not a new carve-out — the same well-tested boundary.

**Still open, deliberately left unsolved:** `bitcast`'s exact syntax/semantics. Judged rare enough in practice (HPC/scientific code needs value conversion constantly, same-width bit reinterpretation rarely) not to be worth designing now.
