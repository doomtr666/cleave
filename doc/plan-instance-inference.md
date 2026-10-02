# Specialization by instance inference

Status: done (2026-10-02). Replaces several mechanisms that each patched the same gap; listed under
"What it replaces", all deleted. As built: see "As built" at the end.

## The gap

Compilation today runs in two disconnected steps: whole-program HM inference gives every generic
function a scheme and types every call from it; monomorphization then *substitutes* concrete types into
bodies typed once, generically. Whatever a body only reveals with concrete types can't flow back to its
callers:

- the type of a field on an unannotated parameter (`fn first(t) { t[0] }`);
- how many copies an unrolled loop has, which branch of an `if` on a const generic is dead;
- an output-only type computed by a body: `impl<Opt, M: Trainable, S> Optimizer<Opt, M, S>`, where `S`
  is whatever `init_state` builds — the reason `NetworkState` has to be written by hand today.

## The idea

Specialize on demand, by inferring each instance, as Julia does (and C++ for `auto` return types):

- An **instance** is a function or impl method at concrete argument types (and concrete impl targets,
  as far as the call determines them). Roots are the non-generic functions and impl methods.
- Inferring an instance infers its body with those concrete types. When the body calls a generic
  function or a generic impl method with arguments that are concrete by then, the callee's instance is
  inferred (recursively, memoized), and **its result type is the call's type**. Output-only types (the
  optimizer state) fall out: the call gets what the callee's body computed.
- Each instance's body is its own: loops over heterogeneous collections are unrolled and folding `if`s
  pruned on a copy of the body, with the instance's concrete types in hand.
- An instance still being inferred (recursion) answers from its HM scheme.

HM inference stays for what it does well: checking generic code once, typing non-generic code, giving
recursive calls a type. Only *specialization* changes, from substitution to inference.

## How it plugs in

- `Infer` records, for every call to a generic function or generic impl method, the callee and the
  call's argument/result types (`instance_calls`).
- Before defaulting and the final checks, an instance's inference resolves these calls through an
  oracle: every call whose argument types are concrete gets its callee instance's result unified into
  its own result type; repeated until nothing changes (one resolution can make another call's
  arguments concrete), then once more after defaulting.
- The oracle is the new monomorphizer: a memo table from instance key to its inferred instance
  (param types, result, node types, call names, body), filled recursively. It produces the same
  `MonomorphizedProgram` CPS conversion reads today.

## What it replaces

In `monomorphize.rs`: `resolved_target_sigs`, the `deferred_impl` retry queue (up to 64 retries, with a
history of order-dependent flakiness), `patch_ordinary_fn_node_types` (patching `main` up to 8 times
after the fact), the duck-typed re-inference of functions, `with_external_state_hint`. In `unroll.rs`:
the whole-program trial-inference rounds and pack templates (an instance of `impl<Ts...: Print>` at a
concrete tuple is an ordinary instance, unrolled on its own copy). `NetworkState`-style hand-written
state structs in user code.

## Migration

1. Build the instance engine beside the current monomorphizer, selected by a switch, producing the same
   `MonomorphizedProgram`.
2. Run the whole suite with it; fix until green; make it the default.
3. Delete what it replaces, one mechanism at a time, the suite green after each.
4. Then the user-facing payoff: `Optimizer` for a `Trainable` struct written once in `nn`, and the MNIST
   kernel without its hand-written impl and `NetworkState`.

## Open questions

- Recursive generic functions whose result type the scheme doesn't determine (rare): fixed point, or
  require an annotation.
- Compile time: an instance per concrete signature, as today; inference per instance instead of
  substitution is heavier, measured as we go.
- Lambdas passed to generic functions: instance key includes the lambda's identity, as today.

## As built

- `infer.rs`: the `InstanceOracle` trait; `Infer` records every call into a generic fn or impl method
  (`instance_calls`) and resolves those whose arguments are concrete in `settle_before_checks`, as a
  fixpoint with deferred field accesses (an instance's answer can type a value a field access waits on,
  and the other way round), before and after defaulting. Declared generics of an instance never take a
  body literal's default (`undefaultable`): left unpinned they are an error, as at the scheme level.
- `monomorphize.rs::InstanceEngine` is the oracle: one inference per instance, memoized, recursive,
  each on its own copy of the body, unrolled and pruned in rounds (`infer_unrolling`). It drives both
  worklists (fns and impl methods); lambdas still specialize by substitution.
- Roots (non-generic fns) are inferred again through the engine. A root whose scheme-level inference
  failed only because a callee's result is unknown from its scheme alone (`let a = f(t); a[1]`) is
  repaired by that inference when it comes out concrete.
- A const generic read as a value (`for b in 0..B`) is typed by its value's type (`i32`), not by the
  value itself, which an instance binds it to (`Ty::Const(2)`): otherwise two loop variables over
  different bounds had different types. The value travels separately (`Infer::const_refs`) and is put
  back for `cps.rs`, which converts it as a literal; `[v; N]` takes its length from the generic itself.
- Lambdas in an instance's body are specialized from the instance's own inference
  (`InstanceEngine::instance_lambdas`): a comprehension's function only exists in that instance's
  copy of the body, so the program-wide inference never saw it.
- The user-facing payoff: `optim.cleave`'s `impl<Opt, M: Trainable, S> Optimizer<Opt, M, S>`, written
  once with comprehensions (`doc/plan-compile-time-sequences.md`, step 3); `nn` marks `Dense`, the
  MNIST kernel marks its `Network` and writes no optimizer code.
