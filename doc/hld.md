# High-Level Design: Math DSL → e-graph → MLIR compiler

*Summary of architecture principles agreed on so far. A reference to avoid re-deriving these decisions later, not an implementation plan.*

**Positioning:** cleave is a converged language for HPC (high-performance computing) in general — CFD, N-body/SPH, linear-algebra-heavy simulation, and ML/neural-network training all sit on the same footing. ML is one HPC workload among many, not a distinct target needing special-casing: a training loop is just another algebra-driven numerical computation, handled by the exact same extensibility/optimization machinery as any other domain.

**Project values:** open source, no vendor lock-in. Initial/reference hardware targets are **CPU** and **Vulkan Compute** (via MLIR's existing `spirv` dialect), not a proprietary single-vendor stack — deliberately chosen over CUDA-only paths despite CUDA's tooling maturity. Matrix-multiply-heavy workloads target the cross-vendor `cooperative_matrix` Vulkan extension (NVIDIA, AMD, Intel, ARM) for hardware matmul acceleration, rather than a vendor-specific tensor-core API. This doesn't close the door on other backends later — the hardware-target plugin mechanism (see "Cost-driven extraction" below) allows adding any target, including CUDA/PTX, as a community-contributed plugin — but **CUDA/PTX is explicitly not shipped as a default/bundled backend**. Proprietary, single-vendor backends are opt-in additions someone else can bear the cost of maintaining, never part of the project's own default scope.

**Scope discipline:** the goal is performance and ergonomics for scientists/HPC programmers — not a formally-verified language. Formal verification of community axioms (see "Soundness/governance" below) is a plausible future extension, deliberately kept out of v1 scope; a program author is trusted the same way a C or Rust programmer is trusted, not asked to prove anything.

## Pipeline overview

```
Pest grammar → surface AST
             → desugar to functional core (CPS)
             → e-graph (egg) : symbolic/algebraic optimization, saturation
             → cost-driven extraction (per target)
             → lowering to MLIR (progressive, dialect by dialect)
             → target-specific codegen (guided by machine description)
             → [optional] distribution: data partitioning across nodes
             → PGO selects among plausible extracted variants
```

## Core philosophy: don't lower prematurely

The central thesis: keep mathematical structure (matrices, complex numbers, domain-specific algebras) alive as first-class, structural nodes for as long as possible. Lowering a matmul into scalar `+`/`*` loops immediately destroys the information needed for high-level algebraic optimization (e.g. reordering `matmul(matmul(a,b),c)` vs `matmul(a,matmul(b,c))` for FLOP count) — once buried in scalar ops, that opportunity is unrecoverable by later passes.

This is why the design consistently prefers **structural representation over opaque leaves**: `Complex([Id;2])` (real/imag as child e-classes) rather than an opaque complex literal, tensor ops as named nodes (à la MLIR's `linalg` dialect) rather than pre-unrolled loops. Structural representation lets existing rules/analyses on the components apply for free.

This is also the mechanism that makes a **general-purpose scientific language** viable rather than "tensors hardcoded, everything else second-class": a community-contributed algebra (e.g. SPH) gets the exact same treatment (stay structural, stay high-level, lower late) as built-in matrices — scientists never need to manually lower their domain into arrays themselves.

Precedent: this mirrors why MLIR itself exists (multi-level IR, gradual lowering) rather than jumping straight to LLVM scalar IR.

## Type system: analysis data, not enum variants

Encoding every numeric type (i8..i128, f16..f128, `Complex<T>`, vectors, matrices, tensors) as a separate `Language` enum variant doesn't scale — combinatorial explosion of variants × operators.

**Approach:** keep a small, generic node set (e.g. one `Add`/`Mul`/generic `Apply` shape); carry **type** (and **shape**, for tensors) as `Analysis` data computed alongside the e-graph, the same way constant folding already works. Rewrite-rule conditions (`is_type`-style closures inspecting `egraph[id].data`) restrict which rule variant fires; where possible, encode type in literal syntax itself (`0` vs `0.0`) for free, condition-less typing.

### Light vs. heavy storage: a trait, not a second kind of type

C# bakes "fits in a register, value semantics" vs. "heap-allocated, reference semantics" into the language as two fundamentally different kinds of type (`struct` vs `class`). Rust already solves this more cleanly without a second kind: one uniform type model, and "cheap to copy / register-friendly" is just the `Copy` marker trait (auto-derivable when every field is `Copy` and there's no `Drop`), not a hardcoded dichotomy.

cleave should follow Rust's approach: "light" (register/SIMD-lane-friendly, e.g. scalars, small fixed-size structs) vs. "heavy" (heap/tensor-backed, explicit layout) is a marker trait — plausibly a zero-axiom `algebra`/`trait` (the same degenerate case already established for law-free declarations), auto-derived where possible from structure rather than always requiring an explicit annotation. This isn't just type-system tidiness: whether a value is light or heavy is exactly the information the cost-driven extraction/machine-description mechanism needs to decide layout (register-resident vs. explicit memory placement) — it feeds the same machinery already central to the design, rather than being a separate concern. It also connects to the scheduling discussion under Distribution: "heavy" values (tensors, values potentially arriving via a future) are precisely the ones a read-latency-hiding scheduler needs to reason about; a "light" value is by definition always immediately available, never a scheduling concern.

### Numeric-fidelity gotchas (real, not hypothetical — hit these during prototyping)

- **Checked arithmetic required** in `Analysis::make` — equality saturation explores "phantom" combinations (via assoc/commute) that don't correspond to real subterms in the source; unchecked arithmetic can overflow-panic on values that never appear in the actual program.
- **`x * 0.0 => 0.0` is unsound for floats** in general (`NaN * 0 = NaN`, `Inf * 0 = NaN`). Don't include this fold without an explicit "no NaN/Inf" contract (mirrors why compilers gate it behind `-ffast-math`).
- **Low-precision folding (f16) must emulate real per-op rounding** — computing in f64 and casting once at the end does not reproduce real f16 arithmetic behavior.
- Floats need an `Ord`-safe wrapper (`ordered_float::OrderedFloat`) since `egg::Language` requires `Eq + Hash + Ord` and raw `f64` has neither (NaN).

## Extensibility: "algebras" as plugins

**Key mapping:** declaring an algebra (a set + operations + axioms, in the math sense) ≈ declaring a `Language` fragment + a `Vec<Rewrite<L,N>>`. Rewrite rules are already composable data — `Runner::run` just takes a slice, so merging core rules with community rules requires no core changes.

**What a plugin must supply:**
1. Operator signatures (name + arity, + type if applicable)
2. Axioms as rewrite rules
3. A fold/evaluation function for constant folding (`Analysis::make` hook) — the one part that's real code, not pure declaration
4. Either an MLIR lowering, or (safer) a definitional expansion into already-lowerable core ops

**The closed-enum problem:** `egg`'s `Language` is a fixed Rust enum, can't be extended by third parties without recompiling the core. Resolution: a generic operator node (egg's own `Other(Symbol, Vec<Id>)` escape hatch, or a custom `Apply(Symbol, Vec<Id>)`) with operator semantics resolved via a runtime registry. Trade-off: e-matching performance (discriminant-based indexing) vs openness.

### Closing the loop: primitive types are algebras too, not a privileged special case

Every extensibility argument above has an implicit asymmetry left in it: `i32`/`f32` and their operations have, so far, been described as hardcoded into the Rust compiler (fold function, MLIR lowering written by hand), while a community algebra (SPH, a custom `Ring`) goes through the plugin mechanism, declared in cleave source. This is exactly the kind of special-casing this design has eliminated everywhere else (a function body is just a definitional axiom, not a separate mechanism; inlining is just an equality; autodiff is just more axioms) — except this one was never actually closed.

**Resolution:** primitive types are declared as `algebra`s too, in cleave source, using the identical mechanism as any community contribution. Since `i32.add` can't be defined via expansion into something more primitive — it *is* the bottom of the stack — the "MLIR lowering" a plugin can supply (point 4 above) needs a concrete, language-level form: a thin intrinsic that says "this operation *is* this specific MLIR op," expressible directly in cleave source rather than only as hand-written Rust compiler code. **Implemented and shipping** (`stdlib/num/num.cleave`, `cleave/src/mlir_lower.rs::lower_raw_mlir_op`) — not via a bodyless attribute as first sketched, but as a real function body whose only content is a reserved `mlir::dialect::op(...)` call, recognized structurally by its path's own first segment rather than algebra-dispatched:

```
algebra Ring<T> {
    fn add(a: T, b: T) -> T;
}
impl Ring<i32> {
    fn add(a: i32, b: i32) -> i32 { mlir::arith::addi(a, b) }
}
```

A real, ordinary function body (not a special bodyless-method form) is what lets a *composite* intrinsic — one whose meaning needs more than a single MLIR op, `abs` built from `subi`+`cmpi`+`select`, say — fall out for free: it's just an ordinary `let`-chain body, each step its own `mlir::...` call, no new sequencing mechanism needed beyond the one leaf-level call form. A static attribute an op needs beyond its bare operands (`arith.cmpi`'s own `predicate`) is a named call-argument whose value is a string literal, carrying the attribute's raw MLIR text verbatim (`mlir::arith::cmpi(a, b, predicate: "2 : i64")`) — scoped to only this one syntactic slot, not a general string type (which would eventually desugar to `i8[]`, kept entirely separate). Type lowering gets the same treatment one level up: `#[mlir_type("i32")]` on the relevant marker `impl` declares a primitive type's own MLIR representation, so `ty_to_mlir` has no per-type-name Rust match left either, beyond `bool` (a genuine structural special case, matching `infer.rs`'s own treatment of it for `if`/`while` conditions).

**Why this matters, concretely:**
- The Rust compiler's actual hardcoded core shrinks to: the parser, the generic e-graph/algebra-processing engine, and *one* generic "emit this named MLIR op" primitive. Everything else — the entire standard numeric library (`i32`, `f32`, `Complex`, `Ring`, tensors, all of it) — is written *in cleave itself*, as algebra declarations, no different in kind from a community-contributed SPH algebra.
- This is the strongest possible validation of the extensibility claim made throughout this document: not just an assertion that community algebras are first-class, but a verifiable fact, since even the standard library gets no hidden privilege.
- Real precedent for this pattern: self-hosted standard libraries built on a thin intrinsics layer (Zig's `@`-builtins, Rust's own `core::intrinsics`) — a small, fixed set of "this literally is a native op" escape hatches, with everything else built on top in the source language itself.

This is a genuinely structuring decision, not a minor cleanup — it's what keeps the actual compiler core small regardless of how large the numeric/domain vocabulary built on top of it grows.

### Soundness/governance — v1 trust model: no proof kernel

**Explicitly out of scope for now:** a full LCF-style proof kernel (contributed axioms verified by a checkable proof term before admission) was considered, but it amounts to building a proof-assistant kernel as a subproject — a different, much larger undertaking than the actual goal (performance + ergonomics for scientific/HPC computing). C doesn't ask a programmer to prove anything about their code beyond it being valid C; cleave's v1 trust model for axioms follows the same posture, deliberately.

1. **An axiom is a trusted assertion by its author, full stop** — exactly like a Rust trait `impl`: the compiler checks it type-checks, never that it's actually true or sensible. A wrong axiom is a bug, not a compile error, same as any other wrong code.
2. **Root-ownership scoping stays** (an "orphan rule" for rewrite rules: a rule may only rewrite, as its LHS root, an operator its own declaring algebra owns, checked statically at registration time) — this isn't a proof system, just a cheap, mechanical, static check, and it's worth keeping because it blocks the cheapest class of bug to prevent (an algebra reaching into foreign territory) without requiring anything to be proven about a rule's content.
3. **`explain_equivalence` is promoted to the primary debugging tool**, not a secondary safety net behind a proof gate that no longer exists — when an axiom turns out to be wrong, this is how you trace back to it, the same role a debugger plays for an ordinary C bug.

Formal verification of contributed axioms (the LCF-style kernel, or automatic decision for decidable theories like ring/field identities) remains a plausible **future extension** — not a v1 requirement.

**Pragmatic middle tier for community submissions, between "trust the author" and formal verification:**

1. **LLM-assisted first-pass review** — cheap, catches obvious/known mistake patterns (wrong operand order, false commutativity claims, forgotten edge cases). Not a soundness guarantee: an LLM has no decision procedure, can miss subtle counterexamples, and — the specific danger given the threat model here — can produce false confidence on a wrong axiom, which is worse than no review since the whole failure mode is an error that already looked correct to its author. Treat as a second reviewer of comparable rigor, not a stronger form of verification.
2. **LLM-generated, actually-executed property-based tests** (à la QuickCheck/proptest) — the more trustworthy use of an LLM here: not asking it to *judge* the axiom, but to *generate* a test harness (random instances + known edge cases: 0, 1, -1, NaN/Inf, empty collections) that gets run for real. This tests actually-computed values rather than plausibility, and is the pragmatic, industry-standard substitute for proof that essentially all real-world software (this compiler included) already relies on — consistent with the "trust the author, C/Rust-level rigor" posture, not a step toward formal verification.

## Cost-driven extraction as the unifying mechanism

`AstSize` stops being a meaningful cost once tensors/hardware targets are involved. The `CostFunction` becomes parameterized by shape (from Analysis) and by a **machine description** (SIMD width, memory hierarchy, accelerator specifics).

Concretely, this is where the structural (not opaque) representation of `MatMul` cashes in for real hardware: on a target whose machine description advertises `cooperative_matrix` support (see project values above), the cost-driven extractor can lower a structural `MatMul` node straight to that Vulkan/SPIR-V primitive instead of a naive shader loop — only possible because the operation was never prematurely unrolled away.

- **"Plausible variants" for PGO fall out for free**: a saturated e-graph already holds many equivalent implementations simultaneously — extracting top-K (not top-1) per target is not a separate generation step.
- **Real edge over ad hoc autotuners** (Halide/TVM-style): all extracted candidates are provably semantically equivalent by construction (traceable via `explain_equivalence`), so PGO only needs to measure speed, never re-verify correctness.
- **Hardware targets fit the same plugin shape as math algebras** (name + cost function + MLIR lowering) — one extension mechanism for the whole system, not two.
- **Inlining decisions are subsumed into extraction too**: for pure calls, union the call node and its substituted body into the same e-class (provably equal); let the same cost-driven extractor decide which form is cheaper after simplification. No separate inlining-heuristic phase needed.

### A function body is a definitional axiom, not a separate mechanism

Sharpens the point above rather than changing it: a pure function's defining equation (`add(a,b) == a+b`) and a genuine algebra axiom (`add(add(a,b),c) == add(a,add(b,c))`) have exactly the same shape in the rewrite-rule set — both are just equations the saturation engine can use, both union the same way. What differs is **proof provenance, not mechanism**: a definitional equation needs no proof at all — it's true by construction, since it's literally what fixes the meaning of the symbol (the classical distinction between *definitional* and *propositional* equality). A genuine algebra axiom is a claim about already-defined operations and must go through the LCF-style proof pipeline (auto-decided for known theories, or an explicit proof term otherwise). Inlining is simply the safest, proof-free end of the same spectrum the soundness/governance section already establishes for axioms — not a third thing. As before, this only holds for *pure* functions; an effectful function's body isn't freely substitutable as an equation, for the same ordering reasons already noted for effects.

**Effects/purity: per-node, not per-function — and pragmatically minimal.** An early draft of this idea proposed contagious impurity (a function calling `print` anywhere becomes "impure" as a whole, à la Haskell's `IO`) — wrong for this design's actual workload: an ML training loop that calls `print(loss)` once per batch would have its *entire* body (all the matmuls, gradients, weight updates) blocked from algebraic optimization, for one logging call touching none of the hot path. Purity must be tracked per e-class (the same Analysis-data mechanism as type/shape/uncertainty), not per function — an effectful call taints only itself and whatever has a real dependency on it, never the enclosing function wholesale.

The one rule that actually matters, kept deliberately minimal: **an effectful call must never be eliminated as dead code just because its return value is unused** — the effect is the point. Recognizing that two `print` calls are *equivalent* (same arguments) is harmless on its own; the risk was only ever in using that equivalence to justify merging/eliminating repeated calls (idempotence is a separate property from equivalence — repeating an effect is observably different from doing it once, even when each individual call is "the same"). Beyond that one rule, this document does *not* prescribe a general ordering mechanism (e.g. implicit resource tokens for shared external state like stdout) — in practice, real ordering needs are overwhelmingly already carried by ordinary data dependencies (loop-carried state feeding what gets printed), and building machinery for the rare case where two effects must stay ordered *despite* no data dependency between them isn't worth doing speculatively. Left as an open risk, not solved here. Memory serialization / cross-thread visibility (concurrency) is a related but distinct concern, deliberately deferred further still.

### Autodiff and uncertainty propagation are also just axioms, not separate mechanisms

Neither autodiff nor numerical-error estimation need dedicated compiler passes — both are ordinary operators (`D(f, x)`, `uncertainty_of(f)`) whose "axioms" happen to be chain-rule-shaped rewrite rules, living in the exact same extensible rule set as everything else:

```
axiom d_add(f, g, x): D(add(f,g), x) == add(D(f,x), D(g,x))
axiom d_mul(f, g, x): D(mul(f,g), x) == add(mul(D(f,x),g), mul(f,D(g,x)))
axiom d_var(x):       D(x, x) == 1
axiom d_const(c, x):  D(c, x) == 0
```

Saturation applies these exactly like any other rule, progressively eliminating `D(...)` nodes by replacing them with ordinary arithmetic on sub-derivatives — this is symbolic differentiation *as* rewriting, not a bolted-on pass.

**A leftover `D(...)`/`uncertainty_of(...)` node in the extracted result is a legible signal, not a crash.** It means either the function genuinely isn't differentiable/error-bounded there, or (more commonly) nobody has supplied a rule for that specific operator yet — the same "partially simplified, contains an unresolved term" outcome already familiar from ordinary saturation on an under-constrained expression. This is exactly why algebra extensibility composes for free here: a community algebra doesn't need to support autodiff to be useful for ordinary computation, but if it wants to, it just adds more `D(my_op(...), x) => ...` rules to the same rule set it already contributes to — no separate API surface. Same story for `uncertainty_of`: a handful of core-operator propagation rules (`+`, `-`, `*`, `/`, maybe `sqrt`) cover the common case on day one, and coverage improves incrementally as more algebras contribute their own rules — never a monolithic system to build upfront. A leftover node surfaces naturally through the `explain`/introspection tool already proposed elsewhere, rather than needing a new diagnostic channel. Whether a leftover node is a hard error or just a "no info available here" depends on the caller: a training loop that explicitly asked for a gradient and got a residual `D(...)` has a real, actionable compile error; an opportunistic `uncertainty_of` estimate with a residual node just means that part's uncertainty isn't known, not that anything failed.

**Naming note:** the propagation operator is named `uncertainty_of`, not anything "error"-flavored — `erf` (the Gauss error function) is already reserved, well-known mathematical vocabulary that a scientific-computing language must support as an ordinary function, and colliding with it would undermine the whole "vocabulary that means something to a mathematician" principle. `uncertainty_of` isn't just collision-avoidance either — "uncertainty propagation" is the precise, standard term for exactly this operation in experimental science (how a derived quantity's uncertainty follows from its inputs'), and it's the *same* first-order Taylor propagation whether the small perturbation being propagated comes from a physical measurement or from floating-point rounding. Reusing existing correct terminology, not inventing new vocabulary.

### Symbolic integration: explicitly limited scope, not attempted in full

Differentiation is total and syntax-directed (every rule only needs the current node's shape plus recursively-combined sub-results) — which is exactly why it fits the axiom/rewrite model above so cleanly. Integration does **not** have this property, for a deep reason, not an implementation gap: some ordinary elementary functions provably have no elementary closed-form antiderivative at all (Liouville's theorem — `∫e^{-x²}dx` is the classic example, and it's *why* `erf` had to be invented as its own named special function rather than expressed in existing vocabulary). The real decision procedure (the Risch algorithm) is one of the notoriously hardest classical algorithms to implement correctly — real CASes (Mathematica, Maple) have historically shipped incomplete or buggy implementations of it. A promised `is_integrable` predicate would need this same machinery to be trustworthy; without it, "no certainty whether something is integrable even with a large rule set" (as observed) is the honest state of affairs, not a bug to fix with more rules.

**Scope for cleave:** a small, best-effort set of syntax-directed integration axioms for the genuinely compositional cases (linearity, the power rule, a handful of known standard integrals) — same graceful degradation as `D`/`uncertainty_of`: a leftover `∫(...)` node just means "couldn't reduce this," no promise of completeness.

**`is_integrable` as a weak, compile-time predicate, not a decidability oracle.** Not a promise to solve the Risch-grade problem above — just a queryable fact about whether *this particular extraction, with the rules currently available,* left a residual `∫(...)` node or fully resolved. This is information the compiler already has for free (the same leftover-node signal surfaced by `explain`); exposing it as a named, `comptime`-evaluable boolean is what actually makes it *useful*: it's the hook a programmer needs to write their own explicit dispatch ("use the closed form if it resolved, otherwise call quadrature") — which is exactly the mechanism the "no silent fallback" rule below requires to be practical, not just a diagnostic report. Honest caveat: this reflects the engine's *current* rule set, not a timeless mathematical fact — code branching on it can take a different path after a future compiler/rule-set update adds more integration axioms, with nothing in the code itself having changed.

**Explicitly rejected: automatic silent fallback to Monte Carlo (or any numerical method) when symbolic integration fails.** This would silently convert an exact symbolic result into a statistical estimate with sampling noise behind an innocuous-looking call — introducing genuine runtime randomness into a language built around "deterministic, AOT, nothing unknown at runtime," hidden exactly the kind of way this design has consistently rejected elsewhere (silent literal defaulting, premature lowering). Numerical integration is a legitimate, higher-priority *separate* feature — deterministic quadrature (trapezoidal, Simpson, Gaussian) first; Monte Carlo, if offered, as one explicit, clearly-named option, never an automatic fallback.

Worth being precise about *which* Monte Carlo, too: naive uniform-sampling MC is a strawman nobody serious ships — its variance blows up whenever the integrand is peaked or unevenly distributed over the domain, wasting most samples where the integrand contributes nothing. The real minimum baseline is **importance sampling**.

**Importance sampling isn't a bolted-on numerical feature — it's another algebra.** A `Distribution`/`Estimator` algebra with `pdf`, `cdf`, `expectation`, `variance` as its operations (and `cdf(dist, x) == ∫ pdf(dist, t) dt` from `-∞` to `x` as one of its axioms — directly reusing the integration algebra above, not a separate system). Importance sampling's justification, the change-of-measure identity `E_p[f] == E_q[f · p/q]`, is itself just another axiom in this algebra, not a special algorithm the compiler needs bespoke support for. Consequence: "how is `q` chosen" isn't a separate design problem to solve — the e-graph already represents every reformulation of the expectation reachable via that axiom (one per candidate `q`) as equivalent forms in the same e-class, and the *same* cost-driven extraction mechanism used everywhere else in this design picks the cheapest one, now guided by estimated variance (itself computable via `uncertainty_of`) instead of FLOPs. Adaptive importance sampling (à la VEGAS, standard in HEP/physics Monte Carlo — a good fit for cleave's stated HPC/physics audience) falls out of the same mechanism, not a separate algorithm to design.

This also cleanly contains the one genuine source of runtime randomness: `pdf`/`cdf`/`expectation`/the change-of-measure axiom are all purely symbolic and deterministic — only the explicit `sample` operation (drawing an actual value) touches a real PRNG, consistent with "explicit, never silent." PRNG/seeding semantics (reproducibility given a seed) remain a genuinely separate, open concern, not addressed here.

**`pdf_of` belongs in the `D`/`uncertainty_of`/`∫` family, not as a primitive the algebra author writes by hand for every case.** Only *primitive* distributions (Normal, Uniform, ...) supply `pdf`/`cdf` directly as part of their `impl` — the base case, analogous to `D(c,x) => 0`. For a *derived* random variable built from operations on those (`Y = X² + 1`), `pdf_of` is computed by chain-rule-shaped propagation, reusing the algebras already built rather than duplicating their logic:

```
axiom pdf_shift(X, c, y):     pdf_of(add(X, c))(y) == pdf_of(X)(y - c)
axiom pdf_scale(X, a, y):     pdf_of(scale(X, a))(y) == pdf_of(X)(y/a) / abs(a)
axiom pdf_transform(X, g, y): pdf_of(g(X))(y) == pdf_of(X)(inverse(g)(y)) * abs(D(inverse(g), y))
```

This isn't a coincidental family resemblance — the change-of-variables formula for a density *literally contains a derivative* (the Jacobian of the inverse transform), so `pdf_transform` genuinely composes with the `D` axioms rather than reimplementing anything. The sum of two independent variables, `pdf_of(add(X, Y))`, is a convolution — `∫ p_X(t)·p_Y(z-t) dt` — which likewise reuses the integration algebra directly rather than needing a third mechanism. Same graceful degradation as the rest of the family: a non-invertible transform or an intractable convolution leaves a residual `pdf_of(...)` node, a legible signal, not a failure.

## Control flow: kept out of the e-graph, via CPS

Vanilla `egg` models pure term equivalence — not suited to raw CFG (`if`/`while`/`for`, effects, non-local exits) directly.

**Resolution:** desugar all CFG surface syntax into a purely functional core *before* it reaches the e-graph. Adopting **CPS** as that core is what makes this clean: every control construct (`if`, loops, calls, "what happens next") becomes the same syntactic shape — function application to a continuation. Consequence: a **single** beta-reduction rewrite rule uniformly covers inlining, loop unrolling, and branch resolution — no three separate ad hoc mechanisms, and no phase-ordering problem between "inline" and "simplify" (inlining is just another equality explored during the same saturation, exactly like the `x*2 - x` case). CPS also makes effect ordering closer to free — sequencing is just "continuation calls next step," no separate effect-token needed.

**What CPS does *not* solve:**
- Bound-variable handling (capture-avoiding substitution) inside an e-graph is a genuine research-frontier gap in vanilla egg (see: "slotted e-graphs"). CPS makes this *more* central, not less, since now if/loops/calls all route through binder-introducing constructs.
- Beta-reducing a **self-recursive** continuation is exactly "unroll one loop iteration" — must stay guarded/bounded or saturation attempts infinite unrolling (same blow-up class as the assoc/commute overflow hit during prototyping). CPS consolidates this into one rule to guard, rather than three.
- **`let mut x = ...` under branches/loops raises the classical reaching-definitions question ("which definition of `x` reaches this use") — not a free consequence of choosing CPS.** Plain, immutable `let` needs none of this (lexical scoping already disambiguates structurally); this only bites for actual reassignment under control flow. But note this is genuinely *lighter* than classical SSA construction, not just relocated: dominance-frontier computation is specifically needed when starting from an *unstructured* CFG that already lost the original if/while/for shape; converting directly from structured surface syntax to CPS via syntax-directed recursion never loses that shape, so join points (shared continuations) are known for free at each branch/loop node, no graph analysis required. The remaining work is a local def-use check, attached as a *condition* on the copy-propagation rewrite rule — see "Constant and copy propagation" below for the settled resolution.

**Closure conversion** (flattening captured free variables into explicit extra parameters — kept structural, not boxed into an opaque env) is the natural next step after CPS, since every continuation is itself a closure. Because the language is **purely statically typed, heavily AOT-compiled, no JIT, no runtime unknowns**, every call target is statically known — so this flattening applies universally, with no runtime closure ABI ever needed anywhere. Higher-order/generic code (e.g. a function generic over "any Ring") is handled by full **monomorphization** at compile time; the e-graph therefore only ever sees fully concrete, first-order terms. Trade-off: code-size growth from monomorphization — accepted given the already-stated "heavy compilation" tolerance.

## Constant and copy propagation: free for plain `let`, conditional for `let mut`

Classical constant propagation is really three sub-cases bundled together; worth being precise about each rather than waving at "the Analysis handles it" — this section went through two corrections in discussion and this is the settled version.

**Plain, immutable `let a = c+d in body` needs no def-use analysis at all.** Lexical scoping already resolves "which definition does this use of `a` refer to" unambiguously and structurally — there's no reaching-definitions question to ask, no SSA-equivalent prerequisite to establish first. Copy propagation here really is just an e-class union, free, exactly as originally claimed.

**The real ambiguity is specific to `let mut` reassignment under branches/loops** — the same name can be (re)bound along different control-flow paths, and "which definition reaches this specific use" is a genuine question, the same one classical SSA construction answers via dominance-frontier analysis and φ-node placement.

**Correction to an earlier draft of this section, which overstated the difficulty:** dominance frontiers are a *CFG* notion — basic blocks, control-flow edges — and Cytron-style algorithms need them specifically because they start from an *unstructured* representation where the original if/while/for structure has already been lowered away, forcing a graph analysis to rediscover where join points are. cleave's pipeline never goes through an unstructured CFG at all: CPS conversion is a direct, syntax-directed recursion straight over the structured if/while/for AST. At each branch/loop node, the conversion already knows the join point — it's literally the shared continuation passed to both branches — with nothing to reconstruct via graph analysis, because the structure was never lost. The real remaining work is narrower: correctly threading the right value of a mutated variable as an explicit continuation argument at each such node during that same syntax-directed recursion — genuine work to get right, but no dominance/dominance-frontier computation involved, because there's no CFG to compute it over.

**Why the standard LLVM-frontend shortcut (`alloca` + `load`/`store`, let `mem2reg` build the SSA afterward) isn't available here.** That trick works for a frontend like Clang because Clang has no algebraic reasoning to do on the code *before* `mem2reg` runs — it just lowers to LLVM IR. cleave's e-graph/algebra layer needs clean, SSA-equivalent values *before* MLIR exists at all, precisely so it can find algebraic simplifications on real values rather than on memory loads/stores (exactly the premature lowering to memory operations the "don't lower prematurely" thesis argues against elsewhere in this document). `mem2reg` runs too late in the pipeline to help the reasoning that has to happen upstream of MLIR. This is a real constraint, not solved by outsourcing to existing MLIR/LLVM tooling — but it doesn't reopen the dominance-frontier question above: the reason that machinery is avoidable still holds (structured syntax in, never an unstructured CFG), it just means cleave has to implement its own lightweight, syntax-directed version rather than borrowing LLVM's.

**Resolution — reuse the conditional-rewrite mechanism already established.** Whatever the CPS-conversion recursion produces, attach a local def-use check as a **condition on the copy-propagation rewrite rule itself**: `a == c+d` fires only where the check confirms this use is unambiguously reached by that definition, using the exact same conditional-rewrite pattern (`is_type`/`is_const`-style condition closures) established at the very start of this document. Real work, but narrowly scoped, and it slots into infrastructure already proven out rather than requiring a new subsystem.

**Dead-code cleanup doesn't need to be rebuilt either.** Once conditional copy propagation correctly substitutes away every valid use, an original `let mut` binding that becomes unused is just an ordinary MLIR value with zero uses once lowered — MLIR/LLVM's existing, mature DCE handles it for free downstream.

Given a use site is confirmed valid this way:

- **Unconditional constant propagation** is subsumed by `Analysis` (see numeric-fidelity section above) — no separate dataflow/worklist pass needed on top, it's an inherent consequence of congruence closure, continuously, not a sweep.
- **Copy propagation** (`let y = x in y + y => x + x`) is an e-class union — a genuine structural advantage e-graphs have over CFG/SSA representations for values already confirmed single-assignment at that use site.
- **Loop-carried "variables"** (the case SSA needs φ-nodes for) are sidestepped by CPS *representationally* — a loop's per-iteration state is an explicit argument to a recursive continuation — but *producing* that correctly-threaded argument from surface-level mutable-loop syntax is the same construction problem, not sidestepped at all.

**The real gap: path/branch-sensitive refinement** — the "Conditional" in LLVM's SCCP (Sparse *Conditional* Constant Propagation). An e-class represents an **unconditional** equality; it has no way to express "`x` is 3 only inside this branch, not globally." This does **not** fall out of the model for free. It requires the CPS-conversion pass itself to give each branch (`then`/`else`) its **own** refined binding of the tested variable, rather than naively reusing the same binding on both sides — only then does ordinary e-class equality pick up the refinement "for free" inside that branch's body. This is an explicit design decision to make at desugaring time, not an emergent property of the model.

Dead-branch elimination itself needs no new machinery once a condition is known constant — it's just another identity rule, exactly like `add-0`/`mul-1` from the very first prototype: `select(true, then_k, else_k) => then_k`.

This directly bears on the inlining/CPS interleaving discussed above: inlining a call with a known-constant argument only cascades into killing a dead branch *inside* the inlined body if that branch already has its own refined binding for the tested variable — otherwise the cascade stops short, silently, with no error to signal it.

## Scaling strategy for e-graph size

Full whole-program saturation is a real scaling risk (combinatorial blow-up from assoc/commute-style rules, already observed even in toy examples).

- **Per-function saturation first** — bounded scope, tractable size. Mirrors LLVM's function-pass/module-pass split and real-world LTO (heavy optimization per compilation unit, lighter cross-unit pass after).
- **Then a global pass** for cross-function sharing (common subexpressions across functions) — likely achievable via canonicalized structural hashing (GVN-style: sort commutative operands, use folded constants) on already-extracted per-function forms, rather than a second full saturation.
- **Caveat:** committing to one extracted form per function before knowing about cross-function equivalence can hide sharing opportunities — same phase-ordering class of problem, one level up. Canonicalization mitigates the common cases; it won't catch deep cross-function algebraic equivalence that isn't visible after per-function canonicalization.

## Interop and deployment model

cleave is **standalone by default** — a cleave program compiles to a complete, self-contained native executable with its own entry point, the same default posture as a Rust crate compiled as a binary. No host language is required. This costs nothing new: MLIR/LLVM already produce standalone executables as a matter of course, given the already-established "heavy AOT, no JIT" posture.

Functions can be explicitly exposed for external consumption:

- **`extern fn ...`** — declares and calls a foreign C-ABI symbol, no ABI string (`extern "C" fn ...`, this doc's own earlier draft) — implemented and shipping: C is implicitly the only ABI target, the declared signature underneath already says everything else needed, so the string was pure ceremony. Real backing implementation is typically a small Rust crate exposing `extern "C" fn` symbols, registered with the JIT/linker by real function pointer (`cleave-rt`, `main.rs`'s own `--run` path) rather than resolved by dynamic symbol name — see `stdlib/io/io.cleave`'s `Print<T>` for the working end-to-end example, including `extern(symbol)`'s own parenthesized override for the case a bare name can't cover (several algebra-impl methods sharing one cleave-level name, each needing a distinct real symbol).
- **The reverse direction — exposing cleave functions themselves for external consumption** (a host language calling *into* compiled cleave) is not implemented yet; the syntax sketched below is provisional, not finalized, and should be revisited to match the bare-`extern`-no-ABI-string convention above once actually built:
  - `extern fn ...` on a cleave-side definition (not just a foreign declaration) — export with the platform C ABI.
  - A `"rust"`-flavored variant, auto-generating a thin idiomatic Rust wrapper (a trait + impl whose methods just forward to the exported C symbol) so a Rust caller sees an ordinary Rust type, never raw FFI — the difference from the plain case is purely how much glue is generated, not the binary calling convention. Needs its own real design pass (how is "give me a Rust wrapper too" spelled, given there's no ABI string left to carry that marker) before it's built, not assumed here.

Because only concrete, monomorphized signatures ever cross this boundary (in either direction), this stays trivially consistent with the total-monomorphization design — no generics-across-FFI problem to solve.

### Default usage model: the ISPC/CUDA pattern, not a "Rust extension"

cleave is meant to be embedded as a specialized, narrowly-scoped compute kernel called *from* a general-purpose host (typically Rust) — the same relationship ISPC has to its C++ host, or a CUDA kernel has to host C++ code. The host owns orchestration, I/O, and its full ecosystem (web servers, serialization, whatever it needs); cleave owns only the hot numerical computation, called into via the thin wrapper above.

This resolves what looks like a hard problem — reaching into Rust's CPU-committed ecosystem (e.g. Rayon, which is inherently a work-stealing-thread-pool abstraction with no meaningful GPU mapping) from inside cleave — by making it moot rather than solving it: cleave never needs to reach into the host's ecosystem, only the reverse. Parallel/heterogeneous primitives (map/reduce/scan) stay cleave's own abstract, structural algebra nodes, lowered per-target by the same cost-driven extraction mechanism as everything else — never borrowed, CPU-committed host libraries — for the identical "don't lower prematurely" reason motivating the rest of the design. (CUDA enforces the same restriction on `__device__` code, and for the same reason: host-side threading/library abstractions don't have meaningful device semantics.)

## Distribution

**cleave is Turing-complete — genuinely dynamic, data-dependent computation and communication patterns are always *expressible*.** Nothing stops writing an irregular, AMR-style algorithm with dynamic, data-dependent communication in cleave. The "quasi-deterministic" framing below describes where *hardware* rewards regularity, not a restriction the language imposes — an irregular pattern is just harder to map efficiently onto interconnects/architectures built for regular, predictable traffic. Performance-portability concern, not an expressiveness limit.

For the common, favorable case — structured-grid/stencil-style simulations with regular domain decomposition — the communication pattern (who talks to whom, how much, how often) is fixed by problem topology and known essentially at compile time, exactly as the numeric core is fully concrete post-monomorphization. This regularity is what makes treating network operations as CPS continuations tractable in the first place: a send/receive is structurally just `call(f, args, k)`, matching the general "effect ordering via continuations" idea already established for CPS.

**Refinement — precisely where the blocking happens.** CPS itself never blocks — invoking a continuation is just a control transfer, instantaneous by construction. The actual blocking point is **reading** a value that hasn't arrived yet (dereferencing a not-yet-resolved future/promise). This reframes the problem as genuine **scheduling**: among several pending, ready-to-run continuations, choose an order such that by the time one of them actually needs to read a given value, it has had time to arrive — structurally identical to out-of-order/dataflow CPU scheduling (issue instructions whose operands are ready). The fix is still to distinguish *pure* continuations (control-flow desugaring — if/loop/call — compiling to plain tail calls, no runtime involved) from continuations crossing a genuinely asynchronous boundary (network I/O), reified as **promise/future** values — heap-allocated, resumable state registered with a scheduler/executor — but the mechanism to design is a *scheduler that hides read-latency by reordering ready work*, not a blocking/non-blocking dichotomy on the call itself. This mirrors how real async/await implementations (Rust, C#, JS) desugar to CPS-shaped state machines *underneath* a Future/Promise/Task abstraction. CPS stays the right structural IR choice for the local optimization core (see "Control flow" section above); the promise/future + scheduler model governs how async-boundary reads are realized at runtime — a distinct, additional concern, not a contradiction.

Communication scheduling itself (overlap with compute, choice of collective algorithm) fits the same cost-driven extraction mechanism as everything else — expected to reuse the tensor shape/type analysis infrastructure, extended with network topology/bandwidth/latency as part of the machine description. Not yet designed in detail; latency-hiding strategy specifically flagged as needing more thought.

## Open risks (honest list, not yet resolved)

- E-graph-with-binders (CPS/closures route everything through substitution) is not a mature, off-the-shelf part of `egg` — real engineering lift, active research area.
- Guarding recursive-continuation beta-reduction (loop unrolling) against blow-up needs a real bounding strategy, not just "add a condition."
- ~~Rule-soundness verification (LCF-style kernel for community algebras)~~ — deliberately deferred out of v1 scope (see "Soundness/governance"); a future extension, not a current risk to track.
- Per-function → global staging's canonicalization strategy needs concrete design (which normal form, what GVN keys) — sketched conceptually only.
- Distribution/partitioning is named as a goal but not designed at all yet.
- Async/distributed continuations need an actual promise/future runtime model (scheduler, resumable state) distinct from the local pure-continuation compilation strategy — not yet designed.
- Latency-hiding (overlapping communication with compute) needs concrete design — flagged as non-trivial, not yet worked through.
- Effect ordering between two effectful calls with no data dependency between them (e.g. two unrelated `print`s that must nonetheless stay in program order) has no mechanism yet beyond "don't DCE an effect" — deliberately not solved, judged rare in practice.
- Memory serialization / cross-thread visibility (concurrency) is a distinct axis from single-thread effect ordering, not addressed at all yet — "always a great pleasure," deferred further out than the rest.
- Branch-refinement discipline (giving `then`/`else` their own refined bindings during CPS conversion) needs concrete design — without it, path-sensitive constant folding silently doesn't happen, with no signal that an optimization opportunity was missed.
- **`let mut` under branches/loops needs a real, local def-use analysis**, encoded as a condition on the copy-propagation rewrite rule — see "Constant and copy propagation" for the settled design. Plain `let` needs none of this. No dominance-frontier/φ-node computation involved: that machinery is specifically for reconstructing join points lost by lowering to an unstructured CFG, which never happens here (CPS conversion is direct, syntax-directed recursion over the structured if/while/for AST, so join points are known for free). Still real work to implement (the def-use check itself), just narrower than this document originally implied.
