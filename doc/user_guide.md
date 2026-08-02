# cleave: A User's Guide

*How to actually write cleave code — structs, algebras, generics, control flow, and how the type checker resolves what you wrote. Complements `hld.md` (why the language is shaped this way) and `type_inference.md` (how the type checker is actually implemented) — this file explains neither of those; it's "how do I write X", not "why does X work this way" or "what does the compiler do internally". Every example in this guide was actually run through the compiler while it was written, not sketched from memory — see "Where cleave is today" for exactly what "run through the compiler" means right now.*

## Where cleave is today

Two things worth knowing before anything else, so the rest of this guide doesn't read as more finished than it is:

1. **There's no codegen or execution yet.** cleave today is a parser and a type checker, nothing past that — no MLIR lowering, no interpreter, no way to actually *run* a program and see `42` printed anywhere. "Compiles" in this guide always means "type-checks", verified with `cargo run -p cleave --bin cleave -- yourfile.cleave --dump-inference-pass`, which prints every expression back annotated with its resolved type. That's the tool used throughout.
2. **The standard library doesn't yet provide real arithmetic.** `stdlib/num/num.cleave` exists and gives you `Num`/`Int`/`Float` (marker algebras used for literal-defaulting checks — see "Type inference and defaulting" below) but *no* `Ring`/`Ord`-style algebra with real `add`/`sub`/`eq`/`lt` for `i32`/`f64`. Until that lands, `+`/`-`/`*`/`<`/`==` on primitive types only work if *your own program* declares the algebra and `impl`s it — every example below that uses an operator on `i32`/`f64` does exactly that, explicitly, with a stub body (`fn add(a, b) { a }`) standing in for what will eventually be a compiler intrinsic. Don't read those stub bodies as "how you'd really implement addition" — they're placeholders so the example type-checks; there is no way to write a *correct* `add` for a primitive type yet, because there's no way to reach the actual hardware operation from cleave source at all today.

## Hello, cleave

```
fn main() -> i32 {
    42
}
```

A function's body is a block: zero or more `;`-terminated statements, followed by one final expression *without* a semicolon — the **tail** — which is the block's own value. `main`'s tail here is `42`, so `main`'s own value is `42`. Get the semicolon wrong and the meaning changes completely:

```
fn main() -> i32 {
    42;
}
```

This does **not** produce `42` — the trailing `;` turns `42` into an ordinary, discarded statement, leaving no tail at all, so the block's value is `()` (unit). This particular file fails to type-check (`main` promises `i32`, the body produces `()` — the friendly case). The unfriendly case is a function whose declared return type happens to also be `()`, where a stray semicolon silently changes what the function produces with no error at all. When something you expected to come out of a function isn't there, check for a semicolon on the last line first.

## Arithmetic needs a declared algebra (for now)

Since this comes up in nearly every example from here on, one canonical version, reused (with small additions) throughout this guide:

```
algebra Ring<T> {
    fn add(a: T, b: T) -> T;
    fn sub(a: T, b: T) -> T;
    fn mul(a: T, b: T) -> T;
    fn neg(a: T) -> T;
}
algebra Ord<T> {
    fn lt(a: T, b: T) -> bool;
    fn eq(a: T, b: T) -> bool;
}
impl Ring<i32> { fn add(a, b) { a } fn sub(a, b) { a } fn mul(a, b) { a } fn neg(a) { a } }
impl Ord<i32> { fn lt(a, b) { true } fn eq(a, b) { true } }
```

`a + b` is sugar for `add(a, b)`; `-a` is sugar for `neg(a)`; `a < b`/`a == b` are `lt(a, b)`/`eq(a, b)`. None of these are built in for any type — they resolve against whichever declared `algebra` provides them, exactly like any other function call. See "Algebras: how operators actually work" below for the full mechanism; this section exists just so the examples that follow don't each have to explain it from scratch.

## Bindings: `let` and `let mut`

```
algebra Ring<T> { fn add(a: T, b: T) -> T; }
impl Ring<i32> { fn add(a, b) { a } }

fn f() -> i32 {
    let a = 1;
    let mut b = 2;
    b = b + a;
    b
}
fn main() -> i32 { f() }
```

- `let` bindings are immutable — no reassignment, ever.
- `let mut` bindings can be reassigned (`b = ...;`, no `let` on the reassignment) but pay a real cost: a `let mut` binding is **never generalized** (see "Generics" below) — its type is pinned once, monomorphically, at its own declaration.
- Type annotations are optional almost everywhere (`let a: i32 = 1;` works too) — the type checker infers what it can from how a value is used.

## Functions, and how their types get inferred

```
algebra Ring<T> { fn add(a: T, b: T) -> T; }
impl Ring<i32> { fn add(a, b) { a } }

fn add_one(x) { x + 1 }
fn main() -> i32 { add_one(5) }
```

No annotations on `add_one` at all — `x`'s type, and the function's own return type, are both inferred from the body. `x + 1` requires `x` to support addition, so `add_one` ends up polymorphic: usable with any type that has a `Ring` impl, not pinned to one specific type. Compare with an annotated version, same behavior:

```
algebra Ring<T> { fn add(a: T, b: T) -> T; }
impl Ring<i32> { fn add(a, b) { a } }

fn add_one(x: i32) -> i32 { x + 1 }
fn main() -> i32 { add_one(5) }
```

Now the signature is part of the source, checked against (not just inferred from) the body — useful as documentation, and as an explicit boundary once you actually want to *restrict* what a function accepts rather than let it stay maximally general.

**Recursion works directly, in any order of declaration, including mutual recursion between two or more functions:**

```
algebra Ord<T> { fn eq(a: T, b: T) -> bool; }
impl Ord<i32> { fn eq(a, b) { true } }
algebra Ring<T> { fn sub(a: T, b: T) -> T; }
impl Ring<i32> { fn sub(a, b) { a } }

fn is_even(n) {
    if n == 0 { true } else { is_odd(n - 1) }
}
fn is_odd(n) {
    if n == 0 { false } else { is_even(n - 1) }
}
fn main() -> bool { is_even(4) }
```

`is_odd` is used inside `is_even`'s own body despite being declared *after* it — declaration order never matters for top-level functions.

## Control flow

```
algebra Ring<T> { fn neg(a: T) -> T; }
algebra Ord<T> { fn lt(a: T, b: T) -> bool; }
impl Ring<i32> { fn neg(a) { a } }
impl Ord<i32> { fn lt(a, b) { true } }

fn abs(x: i32) -> i32 {
    if x < 0 { -x } else { x }
}
fn main() -> i32 { abs(-3) }
```

`if`/`else` is an expression — both branches must produce the same type, and (with an `else`) the whole `if` evaluates to whichever branch actually ran. Without an `else`, or with mismatched branch types, `if` produces `()` instead.

```
algebra Ring<T> { fn add(a: T, b: T) -> T; }
algebra Ord<T> { fn lt(a: T, b: T) -> bool; }
impl Ring<i32> { fn add(a, b) { a } }
impl Ord<i32> { fn lt(a, b) { true } }

fn sum_to(n: i32) -> i32 {
    let mut total = 0;
    for i in 0..n {
        total = total + i;
    };
    total
}
fn main() -> i32 { sum_to(5) }
```

`while`/`for` are always statements in spirit — neither can produce a value the way `if` can (there's no `break value`), so both are always typed `()`. Because of that, a loop used as a statement inside a block needs its own trailing `;`, exactly like any other discarded expression (`for i in 0..n { ... };`, not `for i in 0..n { ... }` with nothing after) — easy to forget, and the parser will tell you plainly if you do (`expected ... comp_op, add_op, ...`, pointing at whatever follows). The idiom for "compute something via a loop" is a `let mut` accumulator, updated inside the loop, read back afterward — exactly `sum_to` above. `while cond { ... }` follows the identical pattern.

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
algebra Ring<T> { fn add(a: T, b: T) -> T; fn mul(a: T, b: T) -> T; }
impl Ring<f64> { fn add(a, b) { a } fn mul(a, b) { a } }

fn magnitude_sq(v: Vec2) -> f64 {
    v.x * v.x + v.y * v.y
}
fn main() -> f64 { magnitude_sq(Vec2(x: 1.0, y: 2.0)) }
```

## Algebras: how operators actually work

`+`, `-`, `*`, `/`, `==`, `<`, `and`, `or`, ... are not built into the language for any particular type. `a + b` is sugar for a plain function call, `add(a, b)`, resolved against a declared `algebra`:

```
struct Vec2 { x: f64, y: f64 }

algebra Ring<T> {
    fn add(a: T, b: T) -> T;
    fn mul(a: T, b: T) -> T;
}
impl Ring<f64> { fn add(a, b) { a } fn mul(a, b) { a } }

impl Ring<Vec2> {
    fn add(a, b) { Vec2(x: a.x + b.x, y: a.y + b.y) }
    fn mul(a, b) { a }
}

fn translate(a: Vec2, b: Vec2) -> Vec2 {
    a + b   // resolves to Ring::add for Vec2
}
fn main() -> Vec2 { translate(Vec2(x: 1.0, y: 2.0), Vec2(x: 3.0, y: 4.0)) }
```

Once a real numeric stdlib exists, `i32`/`f64`/... will support `+` the exact same way — because the standard library declares `Ring`-style algebras and `impl`s them for those types, not because `+` is special-cased anywhere in the compiler. "Adding operator support to a new type" is never a special mechanism — it's exactly the ordinary `impl` written above for `Vec2`.

**An `impl` method's parameters usually don't need annotating** — they're checked against, and default to, the algebra's own declared signature (`fn add(a: T, b: T) -> T;` above). Write the annotation anyway when it makes the code more readable; the type checker treats it as a redundant check against the algebra's own truth, not a second, independent source of it.

**Two algebras declaring the same operation name is an ambiguity, rejected outright, not silently guessed:** if both `algebra Ring<T>` and some other algebra also declare `add`, calling `a + b` (with a type that has an `impl` for both) is a hard compile error — the language has no "pick whichever seems more specific" heuristic. This is deliberate: two genuinely different `add`s on the same type (wrapping vs. saturating arithmetic, say) are both legitimate, and guessing which one you meant would be worse than asking.

**Recursion within your own `impl`, watch the base case:** `impl Ring<Vec2> { fn add(a, b) { a + b } }`, with `a`/`b` typed `Vec2` directly, would be `Ring::add` calling `Ring::add` for the exact same type forever — no base case. This type-checks fine (it's not a type error — nothing about the *types* is wrong), but it's an infinite loop the moment it would actually run. The correct version bottoms out at the *field* level, where `a.x + b.x` dispatches to a *different*, more primitive `impl` (`Ring<f64>`) — exactly what `translate` above already does.

## Inherent impls: methods with no algebra behind them

Not every method needs operator-dispatch machinery behind it — an ordinary method that only one type will ever have doesn't need an `algebra` at all:

```
struct Vec2 { x: f64, y: f64 }
algebra Ring<T> { fn add(a: T, b: T) -> T; fn mul(a: T, b: T) -> T; }
impl Ring<f64> { fn add(a, b) { a } fn mul(a, b) { a } }

impl struct Vec2 {
    fn magnitude_sq(v) -> f64 { v.x * v.x + v.y * v.y }
}

fn f(v: Vec2) -> f64 {
    v.magnitude_sq()
}
fn main() -> f64 { f(Vec2(x: 1.0, y: 2.0)) }
```

The literal `struct` keyword right after `impl` matters — it's what tells the parser this is an ordinary method block, not an algebra `impl` (`impl struct Vec2 { ... }` vs. `impl Ring<Vec2> { ... }` — dropping `struct` changes the meaning entirely, not just the style).

There's no implicit `self` — `v.magnitude_sq()` calls `magnitude_sq` with `v` filling its **first** parameter, an ordinary explicit one. An unannotated first parameter defaults to the enclosing struct's own type, exactly like an algebra impl's own unannotated parameters default to what the algebra declares.

**One real limitation, worth knowing about explicitly rather than hitting by surprise:** an inherent method with no `->` return-type annotation is only usable, at its real inferred type, from *inside its own body* (a self-recursive call). Called from anywhere else, an unannotated inherent method's return type shows up as unresolved. Always write `-> T` on an inherent method whose result another function actually needs, exactly as `magnitude_sq` does above.

## Generics

A function, struct, or algebra can all be generic over a type:

```
fn first<T>(a: T, b: T) -> T { a }

struct Pair<T> {
    a: T,
    b: T,
}

fn f() -> Pair<f64> {
    Pair(a: 1.0, b: 2.0)   // T inferred as f64 from the field values, no annotation needed
}
fn main() -> f64 { f().a }
```

**A generic function is genuinely polymorphic — usable at different types by different callers, each with its own fresh instantiation:**

```
fn identity(x) { x }

fn g() -> i32 {
    let a = identity(1);
    let b = identity(1.5);   // fine -- a completely independent instantiation of `identity`
    a
}
fn main() -> i32 { g() }
```

This is `let`-polymorphism (Hindley-Milner "let generalization"): a name bound via `let` or a top-level `fn` gets to be reused at multiple, unrelated types; a plain function *parameter* holding the same value does not — inside a function body, a parameter's type is pinned for the duration of that one call. `let mut` bindings are also never generalized, regardless of what their value looks like — a `let mut` is always pinned to one concrete type, for soundness reasons (see `type_inference.md`).

**Bounds** restrict a generic to types that implement a given algebra:

```
algebra Ord<T> {
    fn lt(a: T, b: T) -> bool;
}
impl Ord<i32> { fn lt(a, b) { true } }

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

## Const-generics: types parameterized by a value, not just by a type

A struct's generic parameter list can include `const` entries — ordinary compile-time *values* (today: integers or booleans), not types:

```
struct Vector<T, const N: i32> {
    data: [T; N],
}

fn f() -> Vector<f64, 3> {
    Vector::<f64, 3>(data: [1.0, 2.0, 3.0])
}
fn main() -> f64 { f().data[0] }
```

`N` is checked exactly like a type generic — `Vector<f64, 3>` and `Vector<f64, 4>` are genuinely different, incompatible types, the same way `Vector<f64, _>` and `Vector<i32, _>` would be.

## Turbofish: `::<...>`, for when inference needs a hint

Most of the time, a generic argument is inferred from how a value is used and never needs to be written down. When there's nothing to infer it *from* — an empty array, or a bare numeric literal that would otherwise default to the wrong shape — spell it out explicitly with `::<...>`:

```
struct Vector<T, const N: i32> { data: [T; N] }

fn f() -> Vector<f64, 3> {
    Vector::<f64, 3>(data: [1.0, 2.0, 3.0])   // pins T and N explicitly
}

fn g() -> f64 {
    let id = fn(x) { x };
    id::<f64>(1.0)   // pins the lambda's own generic instantiation
}
fn main() -> f64 { g() }
```

Bare `<...>` (no `::`) isn't used for this — `f<T>(x)` would be genuinely ambiguous with a chained comparison (`f < T > x`), the same reason Rust's own turbofish exists.

## Heterogeneous algebras: relating more than one type at once

`algebra Ring<T>` relates one type to itself (`add: T, T -> T`). Some real operations, matrix multiplication chief among them, genuinely need *more* than one type in play at once — the two operands and the result all have different shapes:

```
algebra MatMul<A, B, C> {
    fn mul(a: A, b: B) -> C;
}

struct Matrix<T, const R: i32, const C: i32> {
    values: [T; R, C],
}

impl<T, const N: i32, const M: i32, const K: i32>
    MatMul<Matrix<T,N,M>, Matrix<T,M,K>, Matrix<T,N,K>> {
    fn mul(a, b) { a }   // a stub -- see "Where cleave is today"
}

fn f() -> Matrix<f32, 2, 5> {
    let a = Matrix::<f32, 2, 3>(values: [[1.0,1.0,1.0],[1.0,1.0,1.0]]);
    let b = Matrix::<f32, 3, 5>(values: [[1.0,1.0,1.0,1.0,1.0],[1.0,1.0,1.0,1.0,1.0],[1.0,1.0,1.0,1.0,1.0]]);
    a * b   // Matrix<f32, 2, 5> -- N and K come from a and b respectively; the shared M cancels
}
fn main() -> i32 { 0 }
```

`a * b`'s shape checking is real: swap `b` for one whose first dimension doesn't match `a`'s own second dimension (a genuinely invalid multiplication) and the call is rejected at compile time, not silently accepted with a garbage result type.

## Higher-order functions: passing a function as a value

A parameter (or any type position) can itself be typed as a function:

```
algebra Ring<T> { fn add(a: T, b: T) -> T; }
impl Ring<i32> { fn add(a, b) { a } }

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

algebra Ring<T> {
    fn add(a: T, b: T) -> T;
    fn mul(a: T, b: T) -> T;
}

impl Ring<f64> { fn add(a, b) { a } fn mul(a, b) { a } }

impl Ring<Vec2> {
    fn add(a, b) { Vec2(x: a.x + b.x, y: a.y + b.y) }
    fn mul(a, b) { a }
}

impl struct Vec2 {
    fn magnitude_sq(v) -> f64 { v.x * v.x + v.y * v.y }
}

fn combine<T: Ring>(a: T, b: T) -> T {
    a + b
}

fn main() -> f64 {
    let a = Vec2(x: 1.0, y: 2.0);
    let b = Vec2(x: 3.0, y: 4.0);
    let c = combine(a, b);       // generic over any Ring, instantiated at Vec2 here
    c.magnitude_sq()
}
```

`combine` is written once, generically, and works for `Vec2` here purely because `Vec2` has a `Ring` impl — the same `combine` would work for `i32`, `f64`, or any other type with its own `Ring` impl, with zero changes to `combine` itself. That reuse — write the generic algorithm once, get it for free on every type that implements the right algebra — is the whole point of the algebra mechanism.
