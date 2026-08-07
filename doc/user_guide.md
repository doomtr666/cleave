# cleave: A User's Guide

*How to actually write cleave code — structs, algebras, generics, arrays, control flow, and how the type checker resolves what you wrote. Complements `hld.md` (why the language is shaped this way) and `type_inference.md` (how the type checker is actually implemented) — this file explains neither of those; it's "how do I write X", not "why does X work this way" or "what does the compiler do internally". Every example in this guide is verified against the compiler for real, not sketched from memory — see `cleave/tests/user_guide.rs`, one test per runnable example here, each actually JIT-executed and asserted against its real return value.*

## Where cleave is today

Two things worth knowing before anything else:

1. **Real codegen and execution exist.** `cargo run -p cleave -- yourfile.cleave --dump-mlir` prints the generated MLIR; `--run` JIT-compiles it and actually executes it, printing `main`'s own return value. "Runs" in this guide always means exactly that, verified — not just "type-checks".
2. **A real numeric standard library exists and is in the prelude.** `+`, `-`, `*`, `<`, `==`, and friends just work on `i8`/`i16`/`i32`/`i64`/`f32`/`f64` out of the box — no declaring your own `Ring`/`Ord` algebra first, no stub bodies. `stdlib/num/num.cleave` (arithmetic/comparison) and `stdlib/logic/logic.cleave` (`and`/`or`/`xor`/`implies`/`not` on `bool`) are both loaded automatically, the same way Rust's own `std::prelude` always is.

What's still missing, worth knowing before you go looking for it (see `doc/backlog.md` for the full, current list): no way yet to produce a standalone executable (`--run` is JIT-only); lambdas type-check but can't be JIT-executed yet (closure conversion isn't implemented); dot-method-call syntax (`v.magnitude_sq()`) on an inherent `impl` type-checks but also can't run yet — both are called out explicitly, with a working alternative, where they come up below.

## Hello, cleave

```
fn main() -> i32 {
    42
}
```

`cargo run -p cleave -- hello.cleave --run` prints `main returned: 42`.

A function's body is a block: zero or more `;`-terminated statements, followed by one final expression *without* a semicolon — the **tail** — which is the block's own value. `main`'s tail here is `42`, so `main`'s own value is `42`. Get the semicolon wrong and the meaning changes completely:

```
fn main() -> i32 {
    42;
}
```

This does **not** produce `42` — the trailing `;` turns `42` into an ordinary, discarded statement, leaving no tail at all, so the block's value is `()` (unit). This particular file fails to type-check (`main` promises `i32`, the body produces `()` — the friendly case). The unfriendly case is a function whose declared return type happens to also be `()`, where a stray semicolon silently changes what the function produces with no error at all. When something you expected to come out of a function isn't there, check for a semicolon on the last line first.

## Arithmetic on primitive types

```
fn add_one(x: i32) -> i32 { x + 1 }
fn main() -> i32 { add_one(5) }
```

`add_one(5)` → `6`. `a + b` is sugar for `add(a, b)`; `-a` is sugar for `neg(a)`; `a < b`/`a == b` are `lt(a, b)`/`eq(a, b)` — none of these are built into the language for any particular type. They resolve against a declared `algebra`, exactly like any other function call — see "Algebras: how operators actually work" below for the full mechanism. What makes this section short is that the resolution already exists for every primitive numeric width, shipped in the prelude (`stdlib/num/num.cleave`) — you only need to declare your own `impl` when you introduce a *new* type that should support these operators too (`Vec2`, later in this guide).

## Bindings: `let` and `let mut`

```
fn f() -> i32 {
    let a = 1;
    let mut b = 2;
    b = b + a;
    b
}
fn main() -> i32 { f() }
```

`f()` → `3`.

- `let` bindings are immutable — no reassignment, ever.
- `let mut` bindings can be reassigned (`b = ...;`, no `let` on the reassignment) but pay a real cost: a `let mut` binding is **never generalized** (see "Generics" below) — its type is pinned once, monomorphically, at its own declaration.
- Type annotations are optional almost everywhere (`let a: i32 = 1;` works too) — the type checker infers what it can from how a value is used.

## Functions, and how their types get inferred

```
fn add_one(x) { x + 1 }
fn main() -> i32 { add_one(5) }
```

No annotations on `add_one` at all — `x`'s type, and the function's own return type, are both inferred from the body. `x + 1` requires `x` to support addition, so `add_one` ends up polymorphic: usable with any type that has a `Ring` impl, not pinned to one specific type. Compare with an annotated version, same behavior:

```
fn add_one(x: i32) -> i32 { x + 1 }
fn main() -> i32 { add_one(5) }
```

Now the signature is part of the source, checked against (not just inferred from) the body — useful as documentation, and as an explicit boundary once you actually want to *restrict* what a function accepts rather than let it stay maximally general.

**Recursion works directly, in any order of declaration, including mutual recursion between two or more functions:**

```
fn is_even(n: i32) -> bool {
    if n == 0 { true } else { is_odd(n - 1) }
}
fn is_odd(n: i32) -> bool {
    if n == 0 { false } else { is_even(n - 1) }
}
fn main() -> i32 { if is_even(4) { 1 } else { 0 } }
```

`is_odd` is used inside `is_even`'s own body despite being declared *after* it — declaration order never matters for top-level functions.

## Control flow

```
fn abs(x: i32) -> i32 {
    if x < 0 { -x } else { x }
}
fn main() -> i32 { abs(-3) }
```

`abs(-3)` → `3`. `if`/`else` is an expression — both branches must produce the same type, and (with an `else`) the whole `if` evaluates to whichever branch actually ran. Without an `else`, or with mismatched branch types, `if` produces `()` instead.

```
fn sum_to(n: i32) -> i32 {
    let mut total = 0;
    for i in 0..n {
        total = total + i;
    };
    total
}
fn main() -> i32 { sum_to(5) }
```

`sum_to(5)` → `10`. `while`/`for` are always statements in spirit — neither can produce a value the way `if` can (there's no `break value`, see `doc/backlog.md`), so both are always typed `()`. Because of that, a loop used as a statement inside a block needs its own trailing `;`, exactly like any other discarded expression (`for i in 0..n { ... };`, not `for i in 0..n { ... }` with nothing after) — easy to forget, and the parser will tell you plainly if you do (`expected ... comp_op, add_op, ...`, pointing at whatever follows). The idiom for "compute something via a loop" is a `let mut` accumulator, updated inside the loop, read back afterward — exactly `sum_to` above. `while cond { ... }` follows the identical pattern, and both actually execute correctly end to end (`--run`), not just type-check.

### Boolean logic: `and`/`or`/`xor`/`implies`/`not`

`&&`/`||`/`!` aren't cleave syntax — the propositional-logic keywords are, with standard precedence (`implies` binds loosest, then `and`/`or`/`xor`, then comparison):

```
fn main() -> i32 {
    let a = true;
    let b = false;
    if (a and not b) implies (a or b) { 1 } else { 0 }
}
```

Like arithmetic operators, these are ordinary algebra dispatch under the hood (`stdlib/logic/logic.cleave`, in the prelude) — `a and b` is sugar for `and(a, b)`, `not a` for `not(a)`.

## Structs

```
struct Vec2 {
    x: f64,
    y: f64,
}

fn origin() -> Vec2 {
    Vec2(x: 0.0, y: 0.0)
}
```

Construction is always named-argument call syntax — `Vec2(x: 0.0, y: 0.0)`, never positional, and every field must be named. Field access is the ordinary `.`:

```
struct Vec2 { x: f64, y: f64 }
fn magnitude_sq(v: Vec2) -> f64 { v.x * v.x + v.y * v.y }
fn main() -> i32 {
    if magnitude_sq(Vec2(x: 3.0, y: 4.0)) == 25.0 { 1 } else { 0 }
}
```

A struct is a **stable reference** — passed and returned by identity, mutated in place, never copied field-by-field on assignment (the same design array values use, see "Arrays" below). Fields can be reassigned directly:

```
struct Vec2 { x: f64, y: f64 }
fn main() -> i32 {
    let mut v = Vec2(x: 1.0, y: 2.0);
    v.x = 10.0;
    if v.x + v.y == 12.0 { 1 } else { 0 }
}
```

## Arrays

```
fn main() -> i32 {
    let mut a = [1, 2, 3];
    a[0] = 10;
    a[0] + a[1] + a[2]
}
```

Returns `15`. Array literals (`[1, 2, 3]`), `[value; N]` repetition (`[0; 4]`), reading, and writing (`a[i] = v`) all work, including multi-dimensional arrays — `a[i, j]` is Fortran-style sugar for `a[i][j]`, both desugaring identically (a multi-dim array's own type is always a nested single-dim array, `[[T; C]; R]`, never a separate primitive):

```
fn main() -> i32 {
    let mut grid = [[1, 2, 3], [4, 5, 6]];
    grid[1, 2] = 60;
    grid[0, 0] + grid[1, 2]
}
```

Returns `61`. An array, like a struct, is a stable reference — `a[i] = v` mutates the same underlying storage every other reference to `a` sees, never a copy.

## Algebras: how operators actually work

`+`, `-`, `*`, `/`, `==`, `<`, `and`, `or`, ... are not built into the language for any particular type. `a + b` is sugar for a plain function call, `add(a, b)`, resolved against a declared `algebra` — the exact mechanism `stdlib/num/num.cleave` uses to back `+`/`-`/`*` for every numeric width, `stdlib/logic/logic.cleave` for `and`/`or`/`not` on `bool`, and the mechanism you reach for the moment you want the same operators on your *own* type:

```
struct Vec2 { x: f64, y: f64 }
impl Ring<Vec2> {
    fn add(a, b) { Vec2(x: a.x + b.x, y: a.y + b.y) }
}
fn translate(a: Vec2, b: Vec2) -> Vec2 {
    a + b   // resolves to Ring::add for Vec2
}
fn main() -> i32 {
    let r = translate(Vec2(x: 1.0, y: 2.0), Vec2(x: 3.0, y: 4.0));
    if r.x == 4.0 and r.y == 6.0 { 1 } else { 0 }
}
```

`Ring<T>` (`add`/`sub`/`mul`/`neg`) and `Ord<T>` (`lt`/`le`/`gt`/`ge`/`eq`/`neq`) are already declared by `stdlib/num/num.cleave` — you only need `impl Ring<Vec2> { ... }`, not a fresh `algebra Ring<T> { ... }` declaration, unless you're introducing a genuinely new operator family of your own.

**An `impl` method's parameters usually don't need annotating** — they're checked against, and default to, the algebra's own declared signature. Write the annotation anyway when it makes the code more readable; the type checker treats it as a redundant check against the algebra's own truth, not a second, independent source of it.

**Two algebras declaring the same operation name is an ambiguity, rejected outright, not silently guessed:** if both `Ring<T>` and some other algebra also declare `add`, calling `a + b` (with a type that has an `impl` for both) is a hard compile error — the language has no "pick whichever seems more specific" heuristic. This is deliberate: two genuinely different `add`s on the same type (wrapping vs. saturating arithmetic, say) are both legitimate, and guessing which one you meant would be worse than asking. There's no qualified-call syntax to disambiguate yet either (`doc/backlog.md`) — see "Heterogeneous algebras" below for how `examples/matmul.cleave` routes around exactly this by not naming its own multiplication `mul`.

**Recursion within your own `impl`, watch the base case:** `impl Ring<Vec2> { fn add(a, b) { a + b } }`, with `a`/`b` typed `Vec2` directly, would be `Ring::add` calling `Ring::add` for the exact same type forever — no base case. This type-checks fine (it's not a type error — nothing about the *types* is wrong), but it's an infinite loop the moment it would actually run. The correct version bottoms out at the *field* level, where `a.x + b.x` dispatches to a *different*, more primitive `impl` (`Ring<f64>`, from the prelude) — exactly what `translate` above already does.

## Inherent impls: methods with no algebra behind them

Not every method needs operator-dispatch machinery behind it — an ordinary method that only one type will ever have doesn't need an `algebra` at all:

```
struct Vec2 { x: f64, y: f64 }
impl struct Vec2 {
    fn magnitude_sq(v) -> f64 { v.x * v.x + v.y * v.y }
}
```

The literal `struct` keyword right after `impl` matters — it's what tells the parser this is an ordinary method block, not an algebra `impl` (`impl struct Vec2 { ... }` vs. `impl Ring<Vec2> { ... }` — dropping `struct` changes the meaning entirely, not just the style).

There's no implicit `self` — `v.magnitude_sq()` calls `magnitude_sq` with `v` filling its **first** parameter, an ordinary explicit one. An unannotated first parameter defaults to the enclosing struct's own type, exactly like an algebra impl's own unannotated parameters default to what the algebra declares.

**Real limitation, worth knowing about explicitly rather than hitting by surprise: `v.magnitude_sq()` (dot-call syntax) type-checks but can't be JIT-executed yet** — `cps.rs` has no conversion for it (`doc/backlog.md`). Until that lands, call the method as an ordinary top-level function instead — inherent methods don't gain a bare-name call form automatically either, so declare it as a plain `fn` if you need to actually *run* it:

```
struct Vec2 { x: f64, y: f64 }
fn magnitude_sq(v: Vec2) -> f64 { v.x * v.x + v.y * v.y }
fn main() -> i32 {
    if magnitude_sq(Vec2(x: 1.0, y: 2.0)) == 5.0 { 1 } else { 0 }
}
```

A second, separate limitation: an inherent method with no `->` return-type annotation is only usable, at its real inferred type, from *inside its own body* (a self-recursive call). Called from anywhere else, an unannotated inherent method's return type shows up as unresolved. Always write `-> T` on an inherent method whose result another function actually needs, exactly as `magnitude_sq` does above.

## Generics

A function, struct, or algebra can all be generic over a type:

```
struct Pair<T> { a: T, b: T }
fn f() -> Pair<f64> {
    Pair(a: 1.0, b: 2.0)   // T inferred as f64 from the field values, no annotation needed
}
fn main() -> i32 {
    if f().a == 1.0 { 1 } else { 0 }
}
```

**A generic function is genuinely polymorphic — usable at different types by different callers, each with its own fresh instantiation:**

```
fn identity(x) { x }
fn g() -> i32 {
    let a = identity(1);
    let b = identity(1.5);   // fine -- a completely independent instantiation of `identity`
    if b > 1.0 { a } else { 0 }
}
fn main() -> i32 { g() }
```

This is `let`-polymorphism (Hindley-Milner "let generalization"): a name bound via `let` or a top-level `fn` gets to be reused at multiple, unrelated types; a plain function *parameter* holding the same value does not — inside a function body, a parameter's type is pinned for the duration of that one call. `let mut` bindings are also never generalized, regardless of what their value looks like — a `let mut` is always pinned to one concrete type, for soundness reasons (see `type_inference.md`).

**Bounds** restrict a generic to types that implement a given algebra:

```
fn smaller<T: Ord>(a: T, b: T) -> T {
    if a < b { a } else { b }
}
fn main() -> i32 { smaller(1, 2) }
```

Calling `smaller` with a type that has no `Ord` impl is a compile error (`no impl Ord<...>`), not a runtime one.

**Algebras can bound each other, too** — declaring that any type implementing one algebra must also count as implementing another:

```
algebra Num<T> {}
algebra Int<T> : Num {}

impl Int<i32> {}
// no separate `impl Num<i32>` needed -- Int<i32> already satisfies a Num bound

fn needs_num<T: Num>(x: T) -> T { x }
fn main() -> i32 { needs_num(5) }
```

(`stdlib/num/num.cleave` already declares exactly this `Num`/`Int`/`Float` hierarchy for every primitive width — this is only worth writing yourself for a genuinely new bound family.)

## Const-generics: types parameterized by a value, not just by a type

A struct's generic parameter list can include `const` entries — ordinary compile-time *values* (today: integers or booleans), not types:

```
struct Vector<T, const N: i32> { data: [T; N] }
fn f() -> f64 {
    let v = Vector::<f64, 3>(data: [1.0, 2.0, 3.0]);
    v.data[0] + v.data[1] + v.data[2]
}
fn main() -> i32 {
    if f() == 6.0 { 1 } else { 0 }
}
```

`N` is checked exactly like a type generic — `Vector<f64, 3>` and `Vector<f64, 4>` are genuinely different, incompatible types, the same way `Vector<f64, _>` and `Vector<i32, _>` would be.

## Turbofish: `::<...>`, for when inference needs a hint

Most of the time, a generic argument is inferred from how a value is used and never needs to be written down. When there's nothing to infer it *from* — an empty array, or a bare numeric literal that would otherwise default to the wrong shape — spell it out explicitly with `::<...>`:

```
struct Vector<T, const N: i32> { data: [T; N] }
fn f() -> f64 {
    let v = Vector::<f64, 3>(data: [1.0, 2.0, 3.0]);   // pins T and N explicitly
    v.data[0]
}
fn main() -> i32 {
    if f() == 1.0 { 1 } else { 0 }
}
```

Bare `<...>` (no `::`) isn't used for this — `f<T>(x)` would be genuinely ambiguous with a chained comparison (`f < T > x`), the same reason Rust's own turbofish exists.

## Heterogeneous algebras: relating more than one type at once

`algebra Ring<T>` relates one type to itself (`add: T, T -> T`). Some real operations, matrix multiplication chief among them, genuinely need *more* than one type in play at once — the two operands and the result all have different shapes. This isn't a stub anymore — `examples/matmul.cleave` is a real, JIT-executable, generic matrix-multiply implementation; here's the same shape at a small, concrete size:

```
algebra MatMul<A, B, C> {
    fn matmul(a: A, b: B) -> C;
}

struct Matrix<T: Float, const R: i32, const C: i32> {
    values: [T; R, C],
}

impl<T: Float, const N: i32, const M: i32, const K: i32>
    MatMul<Matrix<T,N,M>, Matrix<T,M,K>, Matrix<T,N,K>> {
    fn matmul(a, b) {
        let mut result = Matrix(values: [[0.0; K]; N]);
        for i in 0..N {
            for j in 0..K {
                let mut sum = 0.0;
                for k in 0..M {
                    sum = sum + a.values[i,k] * b.values[k,j];
                };
                result.values[i,j] = sum;
            };
        };
        result
    }
}

fn main() -> i32 {
    let a = Matrix::<f32, 2, 2>(values: [[1.0, 2.0], [3.0, 4.0]]);
    let b = Matrix::<f32, 2, 2>(values: [[5.0, 6.0], [7.0, 8.0]]);
    let c = matmul(a, b);
    if c.values[0,0] == 19.0 and c.values[0,1] == 22.0
        and c.values[1,0] == 43.0 and c.values[1,1] == 50.0 { 1 } else { 0 }
}
```

`matmul`'s own shape checking is real: swap `b` for one whose first dimension doesn't match `a`'s own second dimension (a genuinely invalid multiplication) and the call is rejected at compile time, not silently accepted with a garbage result type.

**Not named `mul`, and not wired to `*`, on purpose:** `algebras_with_fn` picks a call's candidate algebra by name and arity alone, with no shape disambiguation at that stage — a second algebra also declaring a 2-arg `mul` would make *every* `*` in the program ambiguous, including a genuinely scalar one. No qualified-call syntax exists yet to disambiguate (`doc/backlog.md`), so matrix multiplication stays an ordinary named method, called directly.

## Higher-order functions: passing a function as a value

A parameter (or any type position) can itself be typed as a function:

```
fn apply(f: (i32) -> i32, x: i32) -> i32 {
    f(x)
}
fn g() -> i32 {
    let inc = fn(x) { x + 1 };
    apply(inc, 5)
}
fn main() -> i32 { g() }
```

`(i32) -> i32` — parameter types in parentheses, then `->`, then the return type — is mandatory here (unlike an ordinary `fn`'s own optional return-type annotation): a bare type annotation has no function body nearby to infer a return type *from*.

**This example type-checks but can't be JIT-executed yet** — closure conversion (extracting a lambda's own captures into an explicit record) isn't implemented, `cps.rs` panics on any `Lambda` node it reaches (`doc/backlog.md`). Higher-order functions over *named, top-level* functions aren't affected by this at all — only a lambda *literal* (`fn(x) { ... }`) hits it.

## Type inference and defaulting, briefly

You rarely need to annotate anything — but two rules are worth knowing explicitly, since they occasionally produce a surprising-looking rejection:

- **No implicit conversions, anywhere — including a bare integer literal into a float.** `fn f() -> f64 { 1 }` is a compile error: `1` (no `.`) is an integer-shaped literal, and integers don't implicitly become floats just because the surrounding context wants one. Write `1.0`, or force it explicitly with a suffix: `1:f64`.
- **An unsuffixed, unconstrained numeric literal defaults to `i32` (no `.`) or `f32` (has a `.`)** if nothing else ever pins it to a specific type. This is only a fallback — if something *does* constrain it (a declared return type, an operation with another already-typed value), that wins over the default every time.

```
fn f() -> i32 { 1 }      // fine: 1 defaults to i32, matching the declared return
fn g() -> f64 { 1 }      // rejected: 1 is int-shaped, f64 is not -- write 1.0
fn h() -> f64 { 1.0 }    // fine
```

## Putting it together: a small worked example

```
struct Vec2 {
    x: f64,
    y: f64,
}

impl Ring<Vec2> {
    fn add(a, b) { Vec2(x: a.x + b.x, y: a.y + b.y) }
}

fn magnitude_sq(v: Vec2) -> f64 { v.x * v.x + v.y * v.y }

fn combine<T: Ring>(a: T, b: T) -> T {
    a + b
}

fn main() -> i32 {
    let a = Vec2(x: 1.0, y: 2.0);
    let b = Vec2(x: 3.0, y: 4.0);
    let c = combine(a, b);       // generic over any Ring, instantiated at Vec2 here
    if magnitude_sq(c) == 52.0 { 1 } else { 0 }
}
```

`combine` is written once, generically, and works for `Vec2` here purely because `Vec2` has a `Ring` impl — the same `combine` would work for `i32`, `f64`, or any other type with its own `Ring` impl, with zero changes to `combine` itself. That reuse — write the generic algorithm once, get it for free on every type that implements the right algebra — is the whole point of the algebra mechanism.
