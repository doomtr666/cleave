//! Regression tests for the release-cascade gap found while discussing the
//! affine-ownership plan (`doc/plan-affine-ownership.md`): a struct field
//! whose type is an array of refcounted structs (or, recursively, an array
//! of arrays of them) was never recursed into by `mlir_lower.rs::
//! lower_release_cascade` — confirmed directly, before this fix, via
//! `--dump-cps-optimized` on a real program: `struct Foo { items: [Boxed;
//! 2] }`, sent through an opaque `extern fn` sink so the e-graph can't see
//! through it, emitted exactly one `release`, targeting `Foo` itself —
//! zero for either embedded `Boxed`. Every such struct leaked its array
//! elements on every release, deterministically, no loop required.
//!
//! These tests exercise the *real* AOT pipeline shape (CPS conversion,
//! e-graph optimization, `insert_refcounting`, then `lower_program`) so
//! they see exactly what `--emit-exe`/`--emit-object` would — mirroring
//! `pipeline.rs::build_optimized_cps` step for step, the same discipline
//! `cleave/tests/mlir_lower.rs::lower`'s own doc comment already commits
//! to for its own (refcounting-free) purpose.

use cleave::cps::{collect_mlir_types, collect_struct_schemas, collect_units, convert_program};
use cleave::driver::compile;
use cleave::egraph::optimize_program;
use cleave::escape::escaping_struct_vars;
use cleave::mlir_lower::lower_program;
use cleave::pipeline::check_type_errors;
use cleave::refcount::insert_refcounting;
use cleave::registry::Registry;
use melior::Context;
use melior::dialect::DialectRegistry;
use melior::ir::operation::OperationLike;
use melior::utility::register_all_dialects;

fn context() -> Context {
    let dialect_registry = DialectRegistry::new();
    register_all_dialects(&dialect_registry);
    let context = Context::new();
    context.append_dialect_registry(&dialect_registry);
    context.load_all_available_dialects();
    context
}

/// Compiles `src` through the exact same CPS-to-CPS chain `pipeline.rs::
/// build_optimized_cps` uses (dead-code elimination, e-graph optimization,
/// a second dead-code sweep, escape analysis, `insert_refcounting`), then
/// lowers to MLIR — the real AOT shape, not `cleave/tests/mlir_lower.rs`'s
/// own refcounting-free `lower`, which would never show a `Release` at all.
fn lower_with_refcounting(context: &Context, src: &str) -> String {
    let (result, _sources) = compile(vec![("test.cleave".to_string(), src.to_string())], &[]);
    let program = result.unwrap_or_else(|e| panic!("compile failed: {e:?}"));
    let registry = Registry::build(&program);
    if let Err(diags) = check_type_errors(&program, &registry) {
        panic!("type check failed: {diags:?}");
    }
    let units = collect_units(&program, &registry);
    let cps_program = convert_program(units, None);
    let cps_program = cleave::cps::eliminate_dead_code(cps_program);
    let (cps_program, _) = optimize_program(cps_program, &registry, false);
    let cps_program = cleave::cps::eliminate_dead_code(cps_program);
    let struct_schemas = collect_struct_schemas(&program);
    let mlir_types = collect_mlir_types(&program);
    let escaping = escaping_struct_vars(&cps_program);
    let cps_program = insert_refcounting(cps_program, &struct_schemas, &mlir_types, &escaping);

    let mlir_types = collect_mlir_types(&program);
    let struct_schemas = collect_struct_schemas(&program);
    let module = lower_program(context, &cps_program, &mlir_types, struct_schemas);
    assert!(
        module.as_operation().verify(),
        "generated MLIR module failed verification"
    );
    module.as_operation().to_string()
}

fn count_release_calls(text: &str) -> usize {
    // MLIR's own generic printer renders a `func.call` op as bare
    // `call @sym(...)` at top level but `func.call @sym(...)` once nested
    // inside another op's region (`scf.if`'s own `then`/`else`, here) --
    // same op, two textual spellings. `"call @cleave_release("` alone
    // matches both, since the longer spelling contains it as a substring.
    text.matches("call @cleave_release(").count()
}

/// The exact bug: `Foo`'s own array field embeds two freshly-built `Boxed`
/// values, never reused after construction (no protected-embedding retain
/// needed), sent through an opaque `extern fn` so the e-graph can't inline
/// `Foo` away. Releasing `Foo` must also release both `Boxed`s — three
/// `cleave_release` calls total, not one.
#[test]
fn releasing_a_struct_cascades_into_a_refcounted_array_fields_own_elements() {
    let context = context();
    let src = "struct Boxed { v: i32, tag: [i32; 1] }
    struct Foo { items: [Boxed; 2] }
    fn make_boxed(x: i32) -> Boxed { Boxed(v: x, tag: [x]) }
    extern fn opaque_sink(f: Foo) -> i32;
    fn main() -> i32 {
        let a = make_boxed(1);
        let b = make_boxed(2);
        let f = Foo(items: [a, b]);
        opaque_sink(f)
    }";
    let text = lower_with_refcounting(&context, src);
    assert_eq!(
        count_release_calls(&text),
        3,
        "releasing `Foo` must cascade into both embedded `Boxed` elements \
         (3 releases total: Foo, items[0], items[1]) -- got:\n{text}"
    );
}

// A nested-dimension cascade test (`[[Boxed; 2]; 2]`) is deliberately not
// included here: found, while writing it, that CONSTRUCTING a struct-leaf
// array nested inside another array isn't supported at all today (a
// pre-existing, unrelated limitation -- `mlir_lower.rs`'s own explicit
// `assert!` in `lower_array_construct`, not something this cascade fix
// touches). `push_cascade_array_elements`'s own multi-dimension recursion
// is exercised by direct code review (it's a straightforward index-path
// builder over `dims: &[i64]`, structurally the same shape `flatten_array_
// dims` already uses elsewhere), not by an integration test here, since no
// cleave program can build the input yet. Tracked in `doc/backlog.md`.

/// Control: an array of a *primitive* element type must NOT trigger any
/// cascade machinery at all -- the overwhelmingly common case, and the one
/// `ty_needs_cascade`'s own cheap bail-out exists to keep free of any
/// per-element GEP/dispatch cost.
#[test]
fn an_array_of_primitives_triggers_no_cascade() {
    let context = context();
    let src = "struct Foo { items: [i32; 4] }
    extern fn opaque_sink(f: Foo) -> i32;
    fn main() -> i32 {
        let f = Foo(items: [1, 2, 3, 4]);
        opaque_sink(f)
    }";
    let text = lower_with_refcounting(&context, src);
    assert_eq!(
        count_release_calls(&text),
        1,
        "an array of primitives has nothing to cascade into -- only `Foo` \
         itself should be released -- got:\n{text}"
    );
}

// A struct-field-array-of-*tensors* cascade test is deliberately not
// included either, for the same reason as the nested-dimension one above:
// found, while writing it, that CONSTRUCTING `[Tensor<f32,3>; 2]` at all
// crashes MLIR verification today (`'llvm.store' op operand #0 must be
// LLVM type with size, but got 'tensor<3xf32>'`) -- confirmed pre-existing
// and unrelated to this fix by reproducing it with no release/extern sink
// involved at all, just the bare construction. `push_cascade_leaf`'s own
// tensor-descriptor-extraction branch is unchanged from the working code
// this refactor preserved verbatim (`lower_release_cascade`'s own prior
// tensor-field handling, now reachable from an array slot too) -- it isn't
// new code this fix introduces, just newly reachable from more call
// sites. Tracked in `doc/backlog.md`.
