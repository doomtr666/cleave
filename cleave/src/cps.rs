//! CPS conversion, **Stages 1-5** — see `doc/hld.md`'s own "Control flow:
//! kept out of the e-graph, via CPS" for the settled design this implements
//! (classical CPS, every control construct becomes function application to
//! a continuation; closure conversion is a separate, later pass, not part of
//! this one).
//!
//! Handles literals, variables, `let`/`let mut`, plain (scalar, bare-`Path`)
//! assignment, `if`/`else`, `while`/`for`, field access, struct
//! construction, arrays (literal/`[v; N]` construction, multi-dimensional
//! reads and writes — see "Arrays" below), and calls — both to a bodyless,
//! `#[mlir(...)]`-tagged intrinsic (Appel's *PRIMOP*: straight-line, always
//! returns) and to a real, possibly-recursive function or algebra-impl
//! method (Appel's *APP*: genuinely needs the continuation-passing
//! convention). Not yet handled: field-mutation assignment (`s.x = v`) — see
//! "Arrays" below — and lambdas (need closure conversion first, a separate,
//! later pass this module doesn't implement).
//!
//! ## Arrays
//!
//! Chosen direction (see `doc/backlog.md`): an array is a **stable
//! reference, mutated in place** — `PrimOp::Store` is a real effect, not a
//! functional update — because copying a whole array per element write is a
//! non-starter for HPC, and because a stable reference sidesteps Stage 4's
//! own mutation-threading entirely (an array's own *identity* never changes
//! across a branch/loop, only its contents — nothing to carry as an extra
//! continuation argument the way a reassigned scalar needs). `PrimOp::Array`/
//! `ArrayRepeat` construct a value, `PrimOp::Load`/`Store` read/write one
//! *flat position* — `args` is `[array, index...]`/`[array, index..., value]`,
//! one or more trailing indices, never just one fixed arity.
//!
//! A multi-dimensional access (`a[i,j]`'s own Fortran sugar, indistinguishable
//! after lowering from a literal `a[i][j]` — both desugar to the identical
//! `Index(Index(a,i),j)` shape, and are semantically identical too: this
//! language's own multi-dim array type is always `Array(Array(T,C),R)`, a
//! nested single-dim array, never a separate primitive) is *never* converted
//! as chained single-index `Load`s — `collect_index_chain` walks the whole
//! run of nested `Index` nodes up front and collapses it into *one*
//! multi-index `Load`/`Store`. This matters for correctness, not just
//! efficiency, specifically for a *write*: going through an intermediate
//! single-index `Load` (read "the row", then store into that) would only be
//! sound if `Load` on an array-of-arrays element *aliased* the original
//! storage rather than copying it out — a real, load-bearing representation
//! choice this module never actually commits to, so it's sidestepped
//! entirely: the whole chain resolves to one flat-offset effect, no
//! intermediate array value ever produced. (A read alone has no such hazard
//! either way — collapsed the same way regardless, for consistency and
//! because it's the more efficient shape for a later lowering pass besides.)
//!
//! Field-access assignment (`s.x = v`) is a separate, still entirely open
//! question — not attempted at all (whether a `struct` is itself a "light"
//! value or a "heavy" one requiring its own effectful-field-store treatment
//! is undecided).
//!
//! ## Mutation across control flow
//!
//! A plain reassignment in straight-line code (`x = e;`) is exactly as easy
//! as a fresh `let`: rebind `x` in `env` for whatever comes after it — see
//! `convert_stmts`'s own `StmtKind::Assign` arm. The genuinely new work is
//! only when a mutation happens *inside* a branch or loop body: the value
//! `x` has *after* an `if`, or on the *next* iteration of a loop, depends on
//! which branch ran / isn't known until the previous iteration finished —
//! exactly `hld.md`'s own "let mut under branches/loops needs threading the
//! correct value as an explicit continuation argument", not a dominance-
//! frontier/φ-node construction (the structured `if`/`while`/`for` AST is
//! never lost, so the answer is already syntactically obvious: whatever
//! enclosing-scope names a branch/loop body might reassign).
//! `mutated_free_vars`/`mutated_free_vars_expr` compute exactly that set — a
//! purely syntactic, shadowing-aware walk, no dataflow fixpoint needed — and
//! `ExprKind::If`/`While`/`For` each thread it as extra parameters on their
//! own synthesized join/loop continuation, alongside the "value" parameter
//! already used for control flow itself. Consequently, `env` is no longer
//! just "what's the CVal for a name" — the continuation type itself
//! (`&dyn Fn(CVal, &CEnv) -> CExpr`) carries the *possibly-updated* env
//! forward alongside the value, so code following a mutating branch/loop
//! converts against the right bindings (state-passing-style CPS, the
//! standard technique for modeling `set!`-style mutation in a CPS IR).
//!
//! ## Flattening first
//!
//! CPS conversion needs one uniform view over every fully-concrete,
//! callable thing in the program — today scattered across
//! `callgraph::ProgramInference` (non-generic top-level `fn`s),
//! `monomorphize::MonomorphizedProgram` (generic specializations), and
//! `ItemKind::Impl` directly (non-generic algebra impls, both real-bodied
//! and bodyless intrinsics). `collect_units` merges all three into one
//! `Vec<ConcreteUnit>` — a re-shaping step, no new inference.
//!
//! ## Resolving a call site's own target unit
//!
//! A call already resolved to a *generic* instantiation has a precise,
//! unambiguous answer sitting in that unit's own `call_names` (built by
//! `monomorphize.rs` for exactly this purpose). A call to a *non-generic*
//! top-level `fn` resolves by its own bare name directly. Neither covers a
//! call dispatched to an already-concrete algebra impl (`Ring<f32>::add`) —
//! nothing records *which* impl a dispatch picked, the same "discarded on
//! the spot" gap `monomorphize.rs`'s own module doc describes for generic
//! dispatch. Resolved here the same *kind* of way: `check_no_overlapping_
//! impls` guarantees at most one impl of the one algebra owning a given
//! method name can coherently match a query, so `(bare name, concrete
//! argument types, concrete return type)` is already a unique key —
//! `call_index` maps exactly that to a unit's own name.

use crate::ast::*;
use crate::infer::{ConstValue, Infer, Ty};
use crate::monomorphize;
use crate::registry::Registry;
use std::cell::Cell;
use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;

// ---------------------------------------------------------------- flattening

pub enum UnitBody {
    Real(Block),
    /// A top-level `extern fn` — the C symbol name, which is just the
    /// function's own cleave name (see `ast.rs`'s own `FnDecl::is_extern`
    /// doc comment: no separate attribute argument, nothing to rename).
    /// Carried into `PrimOp::Extern` — see that variant's own doc comment
    /// for why a real C call is a straight-line `LetPrim`, not a `Fix`.
    Extern(String),
    /// `fprime = derive(f);` (`ast.rs`'s own `FnDecl::derivative_of`) — the
    /// base function's own name. No `Block` exists for this unit at all
    /// (there's no cleave-level body to CPS-convert — see `derivative_of`'s
    /// own doc comment): `convert_program` skips a `Derivative`-bodied unit
    /// entirely, same as it already does for `Extern` (`let UnitBody::Real
    /// (body) = &unit.body else { continue };`), *not* because it never
    /// gets a real body at all — unlike `Extern`, it does, just built much
    /// later (`egraph.rs::synthesize_derivatives`, which needs `f`'s own
    /// body *already* CPS-converted first, hence strictly after
    /// `convert_program` returns, not during it).
    Derivative(String),
}

pub struct ConcreteUnit {
    pub name: String,
    pub params: Vec<Param>,
    pub param_types: Vec<Ty>,
    pub result: Ty,
    pub node_types: HashMap<NodeId, Ty>,
    /// `Some((algebra, method))` for a unit built from an algebra-impl
    /// method (concrete or generic-specialized alike) — `None` for a
    /// top-level `fn`, an inherent-impl method, or a lambda unit. Threaded
    /// structurally at construction time, in `collect_units`'s own two
    /// `ItemKind::Impl` branches, where `d.algebra`/`f.name` are already
    /// directly on hand — deliberately *not* something a later pass has to
    /// recover by parsing a unit's own display name back apart (`"Ring::
    /// add<i32>"` is a one-way `format!`, `monomorphize.rs::display_impl_
    /// instantiation`'s own doc comment; no parser for it exists or should).
    /// First consumer: a later e-graph pass matching a call site's own
    /// callee against an `axiom`'s declared algebra/method name directly,
    /// without caring which concrete instantiation this particular unit is.
    pub origin: Option<(String, String)>,
    /// This unit's own resolved mangled-callee-name map, exactly as
    /// `monomorphize.rs` already builds it — empty for a concrete impl
    /// method (no specialization involved, see the module's own doc
    /// comment on `call_index` for how those calls resolve instead).
    pub call_names: HashMap<NodeId, String>,
    /// How many of `params`'/`param_types`' own *leading* entries are
    /// synthesized capture parameters rather than the unit's own originally
    /// declared ones — `0` for every ordinary unit (a ordinary `fn`/impl
    /// method never has any; see `capture_names`'s own doc comment for why
    /// a Stage-A lambda unit's own are always prepended, never interleaved).
    /// Consulted only by Stage B's own callee-specialization pass, to know
    /// how many of a *resolved callable*'s own leading params to splice into
    /// a specialized callee's own signature as its own new leading params
    /// (`build_higher_order_specializations`) — a Stage-B-specialized unit's
    /// own value here is `0` too: nothing downstream needs to treat *it* as
    /// itself further passable as a callable, out of scope for this pass.
    pub capture_count: usize,
    /// **Higher-order calls only** (see the module's own doc comment,
    /// "Stage B: higher-order calls"). One entry per erased, function-typed
    /// parameter this unit's own signature *used to* declare before Stage B
    /// baked one specific passed-in callable directly into it — `String` is
    /// that erased parameter's own original name, `Vec<String>` is the
    /// ordered list of this *same* unit's own leading capture-parameter
    /// names (already present in `params`, above) whose bound `CVal`s
    /// become that callable's own `CVal::Closure::captures` inside this
    /// unit's initial `env` (`convert_program`'s own per-unit setup —
    /// ordinary parameter binding alone never touches a name that isn't in
    /// `params` at all, which the erased parameter deliberately isn't).
    /// Empty for every *ordinary* unit (Stage A fn/impl/lambda alike).
    pub baked_closures: Vec<(String, Vec<String>)>,
    /// **Higher-order calls only.** For a `Call` node inside *this* unit's
    /// own body that Stage B recognized as passing a callable (lambda-bound
    /// or a bare top-level `fn` name) into one of the *callee*'s own
    /// higher-order parameters: the 0-based positions, among that call's
    /// own source-level argument list, of every such callable argument —
    /// `convert_expr`'s own `Call` arm reads this *before* evaluating
    /// arguments normally, since a callable argument doesn't convert to an
    /// ordinary `CVal` at all (its own captures get spliced in instead —
    /// see that arm's own doc comment). `call_names` (above) already
    /// carries *which* specialized callee unit this same call now resolves
    /// to — this only carries *which argument positions* were erased to get
    /// there. Empty for every call that doesn't need this.
    pub higher_order_args: HashMap<NodeId, Vec<usize>>,
    pub body: UnitBody,
}

/// Scans every non-generic algebra `impl` for a `#[mlir_type("...")]`
/// attribute (`ast.rs`'s own `ImplDecl::attrs` doc comment), building the
/// cleave-type-name -> MLIR-type-text map `mlir_lower.rs` needs to lower a
/// primitive type generically (the type-level counterpart to `#[mlir(...)]`/
/// `mlir::...` calls on the operation side) — e.g. `impl Int<i32> {}` tagged
/// `#[mlir_type("i32")]` contributes `"i32" -> "i32"`. Only a single-target,
/// bare-path target (`impl Int<i32>`, not `impl Foo<Bar<i32>>`) is
/// meaningful here; anything else is silently skipped rather than guessed —
/// `mlir_lower.rs`'s own lookup panics clearly if a type it actually needs
/// was never declared this way.
pub fn collect_mlir_types(program: &Program) -> HashMap<String, String> {
    let mut types = HashMap::new();
    for item in &program.items {
        let ItemKind::Impl(d) = &item.kind else { continue };
        let Some(attr) = d.attrs.iter().find(|a| a.name == "mlir_type") else { continue };
        let Some(text) = attr.args.first() else { continue };
        let TypeKind::Path(path, _) = &d.target.kind else { continue };
        types.insert(path.segments.join("::"), text.clone());
    }
    types
}

/// A `struct`'s own declared shape — generic parameter names (positional,
/// matching `Ty::App`'s own convention) and fields in *declaration* order
/// (not necessarily the order a given `StructLit` writes them in, for named
/// construction — `mlir_lower.rs`'s own struct-construction/field-access
/// lowering needs this canonical order, since a struct value's layout has to
/// agree across every construction/access site for the *same* monomorphized
/// type, not just within one literal). Field types are kept as their
/// original AST `Type` -- not yet resolved to a concrete `Ty` -- since
/// resolving a generic field (`real: T`) needs a *specific* instantiation's
/// own type arguments, known only at each individual use site
/// (`mlir_lower.rs::struct_field_types` does that resolution, mirroring
/// `infer.rs`'s own `zip_struct_generics`/`ty_from_ast_mapped` but without
/// the type-checker's own side effects, which no longer apply this late).
pub struct StructSchema {
    pub generics: Vec<String>,
    pub fields: Vec<(String, Type)>,
}

/// Scans every `struct` declaration in the program — the whole-program
/// counterpart `mlir_lower.rs` needs to lower a struct type/construct a
/// value/access a field generically, the same "no per-struct Rust
/// knowledge" posture `collect_mlir_types` already gives primitives.
pub fn collect_struct_schemas(program: &Program) -> HashMap<String, StructSchema> {
    let mut schemas = HashMap::new();
    for item in &program.items {
        let ItemKind::Struct(d) = &item.kind else { continue };
        let generics = d
            .generics
            .iter()
            .map(|g| match g {
                GenericParam::Type { name, .. } => name.clone(),
                GenericParam::Const { name, .. } => name.clone(),
            })
            .collect();
        let fields = d.fields.iter().map(|f| (f.name.clone(), f.ty.clone())).collect();
        schemas.insert(d.name.clone(), StructSchema { generics, fields });
    }
    schemas
}

/// Builds one `ConcreteUnit` per fully-concrete top-level `fn`/algebra-impl
/// method reachable in the program — see the module's own doc comment.
pub fn collect_units(program: &Program, registry: &Registry) -> Vec<ConcreteUnit> {
    let (mono, program_inference) = monomorphize::monomorphize(program, registry);
    let mut units = Vec::new();

    for item in &program.items {
        match &item.kind {
            ItemKind::Fn(f) => {
                let Some(Ok(fn_result)) = program_inference.results.get(&f.name) else { continue };
                let keys = mono.specializations_of(&f.name);
                if keys.is_empty() {
                    // Non-generic. A bodyless top-level `fn` is rejected by
                    // `callgraph::infer_program` itself (`MissingFnBody`) --
                    // `results` would hold an `Err` there, not `Ok` -- with
                    // exactly two exceptions: `f.is_extern` (see `ast.rs`'s
                    // own `FnDecl::is_extern` doc comment) and `f.
                    // derivative_of` (`fprime = derive(f);`), both of which
                    // `callgraph.rs` accepts bodyless on purpose. So `f.body`
                    // is `Some` at this point unless one of those two.
                    let body = match &f.body {
                        Some(b) => UnitBody::Real(b.clone()),
                        None if f.is_extern => UnitBody::Extern(f.extern_symbol.clone().unwrap_or_else(|| f.name.clone())),
                        None if f.derivative_of.is_some() => UnitBody::Derivative(f.derivative_of.clone().unwrap()),
                        None => continue,
                    };
                    units.push(ConcreteUnit {
                        name: f.name.clone(),
                        params: f.params.clone(),
                        param_types: fn_result.param_types.clone(),
                        result: fn_result.result.clone(),
                        node_types: program_inference.node_types.clone(),
                        call_names: mono.seed_call_names().clone(),
                        origin: None,
                        capture_count: 0,
                        baked_closures: Vec::new(),
                        higher_order_args: HashMap::new(),
                        body,
                    });
                } else {
                    for key in keys {
                        units.push(ConcreteUnit {
                            name: key.clone(),
                            params: mono.params(key).to_vec(),
                            param_types: mono.param_types(key).to_vec(),
                            result: mono.result(key).clone(),
                            node_types: mono.node_types(key).clone(),
                            call_names: mono.call_names(key).clone(),
                            origin: None,
                            capture_count: 0,
                            baked_closures: Vec::new(),
                            higher_order_args: HashMap::new(),
                            body: UnitBody::Real(mono.body(key).clone()),
                        });
                    }
                }
            }
            ItemKind::Impl(d)
                if d.generics.is_empty() && !registry.generics(&d.algebra).iter().any(|g| matches!(g, GenericParam::Const { .. })) =>
            {
                // Non-generic algebra impl — mirrors `monomorphize.rs`'s own
                // `dump_concrete_impl`, re-inferring directly (no template
                // needed, it's already fully concrete) rather than reusing
                // that function's own rendering logic.
                //
                // The *algebra's* own const generics are checked too, not
                // just the impl's (`d.generics.is_empty()` alone, this
                // guard's original condition) — an impl declaring zero
                // generics of its own (`impl Sum<i32> { ... }`) can still
                // inherit a free variable from the algebra's own const
                // generic (`algebra Sum<T, const N: i32> { fn total(x: [T;
                // N]) -> T; }` — `N` is never fixed by which impl matched,
                // only by the call site), found by direct testing: without
                // this, `infer_impl_fn_generic_with_env` below re-infers the
                // method in isolation, with no call site to pin `N` down,
                // leaving it a permanently unresolved var that later panics
                // in `mlir_lower.rs` — the *correct*, fully-substituted
                // specialization already exists in `mono`, built by
                // `monomorphize.rs`'s own `impl_worklist` (see `ImplTemplate::
                // is_generic`'s own doc comment for the identical fix there),
                // simply never reached because this branch matched first.
                let all_targets: Vec<Type> =
                    std::iter::once(d.target.clone()).chain(d.extra_targets.iter().cloned()).collect();
                for f in &d.fns {
                    let mut infer = Infer::new(registry);
                    let Ok(ret) =
                        infer.infer_impl_fn_generic_with_env(&program_inference.global_env, &d.algebra, &d.generics, &all_targets, f, item.span)
                    else {
                        continue; // already reported via --dump-inference-pass
                    };
                    // `infer_impl_fn_generic_with_env` above already rejects
                    // a bodyless, non-`extern` method (`MissingIntrinsic
                    // Attribute`) -- an intrinsic operation always has a
                    // real body now (a reserved `mlir::...` call), so `None`
                    // only ever means `extern` by the time this is reached.
                    let body = match &f.body {
                        Some(b) => UnitBody::Real(b.clone()),
                        None => UnitBody::Extern(f.extern_symbol.clone().unwrap_or_else(|| f.name.clone())),
                    };
                    let targets_str = infer.target_types.iter().map(Ty::to_string).collect::<Vec<_>>().join(", ");
                    units.push(ConcreteUnit {
                        name: format!("{}::{}<{targets_str}>", d.algebra, f.name),
                        params: f.params.clone(),
                        param_types: infer.param_types.clone(),
                        result: ret,
                        node_types: infer.node_types.clone(),
                        call_names: HashMap::new(),
                        origin: Some((d.algebra.clone(), f.name.clone())),
                        capture_count: 0,
                        baked_closures: Vec::new(),
                        higher_order_args: HashMap::new(),
                        body,
                    });
                }
            }
            ItemKind::Impl(d) => {
                // Generic algebra impl — every specialization actually
                // reached, already built by `monomorphize`.
                for f in &d.fns {
                    // A bodyless method (`extern(...)`, or bare `extern fn`)
                    // needs `UnitBody::Extern`, not `UnitBody::Real` — the
                    // same distinction the non-generic impl branch above and
                    // the top-level `fn` branch both already make.
                    // `monomorphize.rs::build_impl_templates` defaults a
                    // bodyless method's own template body to an empty
                    // `Block` (nothing to substitute), so `mono.body(key)`
                    // would otherwise silently produce a real unit with a
                    // trivially empty body instead of a real extern call —
                    // found by direct testing, the first generic *and*
                    // extern-backed impl this codebase ever declared
                    // (`impl<const N: i32> Print<[i8; N]>`).
                    let origin_key = format!("{}::{}", d.algebra, f.name);
                    for key in mono.specializations_of(&origin_key) {
                        let body = match &f.body {
                            Some(_) => UnitBody::Real(mono.body(key).clone()),
                            None => UnitBody::Extern(f.extern_symbol.clone().unwrap_or_else(|| f.name.clone())),
                        };
                        units.push(ConcreteUnit {
                            name: key.clone(),
                            params: mono.params(key).to_vec(),
                            param_types: mono.param_types(key).to_vec(),
                            result: mono.result(key).clone(),
                            node_types: mono.node_types(key).clone(),
                            call_names: mono.call_names(key).clone(),
                            origin: Some((d.algebra.clone(), f.name.clone())),
                            capture_count: 0,
                            baked_closures: Vec::new(),
                            higher_order_args: HashMap::new(),
                            body,
                        });
                    }
                }
            }
            ItemKind::InherentImpl(d) if d.generics.is_empty() => {
                // Non-generic inherent impl — either a concrete struct with
                // no generics of its own, or a generic struct impl'd at one
                // specific instantiation (`impl Vec2<f64> { ... }`). Mirrors
                // the non-generic-algebra-impl branch above (re-infer
                // directly, no template needed) but through `infer_
                // inherent_impl_block` instead of a per-method call — one
                // shared `Infer` across every method of *this* impl block
                // gives real mutual recursion between sibling methods for
                // free (`w.dec().is_odd()` calling back into a sibling
                // `is_even`), the same way `callgraph::infer_program`
                // already does for a mutually-recursive top-level `fn`
                // group.
                let TypeKind::Path(p, _) = &d.target.kind else { continue };
                let struct_name = p.segments.join("::");
                let mut infer = Infer::new(registry);
                let (_, results) = infer.infer_inherent_impl_block(&program_inference.global_env, &d.generics, &d.target, &d.fns, item.span);
                for f in &d.fns {
                    let Some(Ok((param_types, result))) = results.get(&f.name) else { continue }; // already reported via --dump-inference-pass
                    // A bodyless inherent method has no `#[mlir(...)]`/
                    // `extern`-style intrinsic equivalent yet — nothing to
                    // build a unit from.
                    let Some(body) = &f.body else { continue };
                    units.push(ConcreteUnit {
                        name: format!("{struct_name}::{}", f.name),
                        params: f.params.clone(),
                        param_types: param_types.clone(),
                        result: result.clone(),
                        node_types: infer.node_types.clone(),
                        call_names: HashMap::new(),
                        origin: None,
                        capture_count: 0,
                        baked_closures: Vec::new(),
                        higher_order_args: HashMap::new(),
                        body: UnitBody::Real(body.clone()),
                    });
                }
            }
            ItemKind::InherentImpl(d) => {
                // Generic inherent impl — every specialization actually
                // reached, already built by `monomorphize`'s own inherent-
                // method worklist.
                let TypeKind::Path(p, _) = &d.target.kind else { continue };
                let struct_name = p.segments.join("::");
                for f in &d.fns {
                    let origin_key = format!("{struct_name}::{}", f.name);
                    for key in mono.specializations_of(&origin_key) {
                        units.push(ConcreteUnit {
                            name: key.clone(),
                            params: mono.params(key).to_vec(),
                            param_types: mono.param_types(key).to_vec(),
                            result: mono.result(key).clone(),
                            node_types: mono.node_types(key).clone(),
                            call_names: mono.call_names(key).clone(),
                            origin: None,
                            capture_count: 0,
                            baked_closures: Vec::new(),
                            higher_order_args: HashMap::new(),
                            body: UnitBody::Real(mono.body(key).clone()),
                        });
                    }
                }
            }
            _ => {}
        }
    }

    // Every `let`-bound lambda's own specialization, discovered and built by
    // `monomorphize.rs`'s own lambda worklist — one `ConcreteUnit` per
    // concrete instantiation actually reached, exactly the treatment a
    // generic top-level `fn`/algebra-impl method already gets above. Unlike
    // those, a lambda's own `params`/`param_types` here are *widened*:
    // `lambda_free_vars` (this module's own capture analysis, run once per
    // specialization — see its own doc comment for why that's safe despite
    // no shared cache) prepends one leading parameter per free variable the
    // lambda's own body references from its enclosing scope, in sorted-name
    // order — the exact same order `convert_stmts`'s own `StmtKind::Let`
    // arm gathers each capture's own `CVal` in when it builds a `CVal::
    // Closure` for this same lambda, so positions line up at every call
    // site. Not iterated via `program.items` (a lambda has no top-level
    // item of its own) — `program_inference.lambda_schemes`' own keys are
    // every scheme-bearing (i.e. `let`, non-`let mut`-bound) lambda `NodeId`
    // in the whole program, reached or not; `specializations_of` on each
    // one's own mangled origin is empty (not missing) for one never actually
    // called from any concrete entry point, same convention as an unreached
    // generic `fn`.
    for lambda_id in program_inference.lambda_schemes.keys().copied() {
        let origin = format!("<lambda#{}>", lambda_id.0);
        for key in mono.specializations_of(&origin) {
            let own_params = mono.params(key);
            let body = mono.body(key);
            let node_types = mono.node_types(key);
            let captures = lambda_free_vars(own_params, body, node_types);
            let capture_names = sorted_capture_names(&captures);

            let mut params: Vec<Param> = capture_names.iter().map(|n| Param { name: n.clone(), ty: None }).collect();
            params.extend(own_params.iter().cloned());
            let mut param_types: Vec<Ty> = capture_names.iter().map(|n| captures[n].clone()).collect();
            param_types.extend(mono.param_types(key).iter().cloned());

            units.push(ConcreteUnit {
                name: key.clone(),
                params,
                param_types,
                result: mono.result(key).clone(),
                node_types: node_types.clone(),
                call_names: mono.call_names(key).clone(),
                origin: None,
                capture_count: capture_names.len(),
                baked_closures: Vec::new(),
                higher_order_args: HashMap::new(),
                body: UnitBody::Real(body.clone()),
            });
        }
    }

    build_higher_order_specializations(&mut units);

    units
}

/// **Stage B: higher-order calls.** `apply(inc, 5)` — `apply`'s own
/// declared signature (`f: (i32) -> i32, x: i32`) is fully concrete (no
/// `monomorphize.rs`-visible type variable anywhere in it: `(i32) -> i32`
/// is just an ordinary type), so `apply` gets exactly *one* `ConcreteUnit`
/// from the rest of `collect_units` above, same as any other non-generic
/// top-level `fn` — nothing about *that* unit says anything about which
/// concrete callable `f` is bound to at any particular call site. Every
/// caller passing a *different* callable needs its own specialized copy of
/// `apply`'s own body, with `f` "erased" (dropped from the signature
/// entirely — a compile-time-known callable is never a runtime value) and
/// every call to it *inside* `apply`'s own body redirected straight to
/// that specific callable's own unit — the exact same "reverse-derive an
/// instantiation, build one specialization per distinct one reached"
/// discipline `monomorphize.rs` already uses for an ordinary generic `fn`,
/// just keyed by "which callable" instead of "which type", and living here
/// (not there) since it needs `ConcreteUnit`/`CVal` concepts monomorphize.rs
/// doesn't have.
///
/// Two passes, not a full worklist to a fixpoint (out of scope for this
/// pass — a *nested* higher-order call, one only reachable from *inside* a
/// just-built specialization, isn't discovered; flag as a follow-up if a
/// real program hits it):
///
/// 1. **Detect.** Walk every already-collected unit's own body (fn/impl/
///    lambda alike — the *caller* can be any of them) for a `Call` whose
///    argument list includes at least one `Ty::Fn`-typed, `Path`-shaped
///    argument that resolves to a known callable — either a lambda-bound
///    name (`u.call_names`, already populated for exactly this by
///    `monomorphize.rs`'s own extended argument-scanning — see
///    `derive_value_instantiation`'s own doc comment) or a bare top-level
///    `fn`'s own unit name directly. The call's own *callee* is resolved
///    the same three-tier way `resolve_call` itself would (mirrored here,
///    not reused, since there's no `Ctx` yet at this point in `collect_
///    units` — units are still being assembled).
/// 2. **Specialize.** For each distinct `(callee unit, [erased position ->
///    resolved callable])` combination actually found (memoized — two call
///    sites passing the *same* callable to the *same* callee share one
///    specialization; two passing *different* callables each get their
///    own, the "two distinct callables" verification case), build one new
///    `ConcreteUnit`: its own params = every resolved callable's own
///    leading capture params (`ConcreteUnit::capture_count`), in erased-
///    position order, followed by the callee's own *remaining* (non-
///    erased) params unchanged — then every call inside the callee's own
///    body whose bare name is exactly an erased parameter's own original
///    name gets a `call_names` override pointing straight at the resolved
///    callable's own unit, plus a `baked_closures` entry so `convert_
///    program`'s own per-unit setup binds that erased name to a `CVal::
///    Closure` over the freshly-spliced-in capture params (see `Concrete
///    Unit::baked_closures`'s own doc comment for why both pieces are
///    needed together). Every *caller* of a redirected call gets its own
///    `call_names` entry overridden to the specialized unit, plus a
///    `higher_order_args` entry recording which argument positions
///    `convert_expr`'s own `Call` arm must splice captures into instead of
///    evaluating normally.
fn build_higher_order_specializations(units: &mut Vec<ConcreteUnit>) {
    let call_index = build_call_index(units);
    let unit_index: HashMap<String, usize> = units.iter().enumerate().map(|(i, u)| (u.name.clone(), i)).collect();

    struct HigherOrderCall {
        caller_idx: usize,
        call_id: NodeId,
        callee_unit_name: String,
        /// `(argument position, resolved callable's own unit name)`, sorted
        /// by position — the memoization/display key both need a
        /// deterministic order.
        erased: Vec<(usize, String)>,
    }

    let mut found: Vec<HigherOrderCall> = Vec::new();
    for (caller_idx, u) in units.iter().enumerate() {
        let UnitBody::Real(body) = &u.body else { continue };
        let mut exprs = Vec::new();
        monomorphize::collect_exprs_block(body, &mut exprs);
        for e in exprs {
            let ExprKind::Call(path, _, args, ..) = &e.kind else { continue };
            let mut erased = Vec::new();
            for (i, a) in args.iter().enumerate() {
                if !matches!(u.node_types.get(&a.id), Some(Ty::Fn(..))) {
                    continue;
                }
                let resolved = if let Some(mangled) = u.call_names.get(&a.id) {
                    Some(mangled.clone())
                } else if let ExprKind::Path(p) = &a.kind {
                    let bare = p.segments.join("::");
                    unit_index.contains_key(&bare).then_some(bare)
                } else {
                    None
                };
                if let Some(callable) = resolved {
                    erased.push((i, callable));
                }
            }
            if erased.is_empty() {
                continue;
            }
            let name = path.segments.join("::");
            let callee_unit_name = if let Some(mangled) = u.call_names.get(&e.id) {
                mangled.clone()
            } else if unit_index.contains_key(&name) {
                name.clone()
            } else {
                let arg_tys: Vec<String> = args.iter().map(|a| u.node_types[&a.id].to_string()).collect();
                let ret_ty = u.node_types[&e.id].to_string();
                match call_index.get(&(name.clone(), arg_tys, ret_ty)) {
                    Some(n) => n.clone(),
                    // Can't resolve the callee itself at all -- not this
                    // pass's problem to diagnose; leave it for `resolve_
                    // call`'s own panic once real conversion reaches it.
                    None => continue,
                }
            };
            found.push(HigherOrderCall { caller_idx, call_id: e.id, callee_unit_name, erased });
        }
    }

    if found.is_empty() {
        return;
    }

    let mut specialized: HashMap<(String, Vec<(usize, String)>), String> = HashMap::new();
    let mut new_units: Vec<ConcreteUnit> = Vec::new();

    for call in &found {
        let key = (call.callee_unit_name.clone(), call.erased.clone());
        if specialized.contains_key(&key) {
            continue;
        }
        let Some(&callee_idx) = unit_index.get(&call.callee_unit_name) else { continue };
        let UnitBody::Real(callee_body) = &units[callee_idx].body else { continue };
        let callee_body = callee_body.clone();
        let callee_params = units[callee_idx].params.clone();
        let callee_param_types = units[callee_idx].param_types.clone();
        let callee_result = units[callee_idx].result.clone();
        let callee_node_types = units[callee_idx].node_types.clone();
        let mut inner_call_names = units[callee_idx].call_names.clone();

        let erased_positions: HashSet<usize> = call.erased.iter().map(|(i, _)| *i).collect();

        let mut new_params: Vec<Param> = Vec::new();
        let mut new_param_types: Vec<Ty> = Vec::new();
        let mut baked_closures: Vec<(String, Vec<String>)> = Vec::new();

        for (pos, resolved_name) in &call.erased {
            let erased_param_name = callee_params[*pos].name.clone();
            let Some(&resolved_idx) = unit_index.get(resolved_name) else { continue };
            let resolved_unit = &units[resolved_idx];
            let n = resolved_unit.capture_count;
            let capture_params: Vec<Param> = resolved_unit.params[..n].to_vec();
            let capture_types: Vec<Ty> = resolved_unit.param_types[..n].to_vec();
            let capture_names: Vec<String> = capture_params.iter().map(|p| p.name.clone()).collect();
            new_params.extend(capture_params);
            new_param_types.extend(capture_types);
            baked_closures.push((erased_param_name.clone(), capture_names));

            // Every call inside the callee's own (shared, unmodified) body
            // whose bare callee name is exactly this erased parameter's own
            // name resolves directly to the resolved callable's own unit --
            // reusing `call_names`'s own shape exactly, one override entry
            // per such call node (there could be more than one, if the
            // callee calls its own higher-order parameter more than once).
            let mut inner_exprs = Vec::new();
            monomorphize::collect_exprs_block(&callee_body, &mut inner_exprs);
            for ie in inner_exprs {
                if let ExprKind::Call(ipath, ..) = &ie.kind {
                    if ipath.segments.join("::") == erased_param_name {
                        inner_call_names.insert(ie.id, resolved_name.clone());
                    }
                }
            }
        }

        for (i, p) in callee_params.iter().enumerate() {
            if !erased_positions.contains(&i) {
                new_params.push(p.clone());
                new_param_types.push(callee_param_types[i].clone());
            }
        }

        let subs: Vec<String> = call.erased.iter().map(|(i, n)| format!("{}={}", callee_params[*i].name, n)).collect();
        let specialized_name = format!("{}[{}]", call.callee_unit_name, subs.join(","));

        new_units.push(ConcreteUnit {
            name: specialized_name.clone(),
            params: new_params,
            param_types: new_param_types,
            result: callee_result,
            node_types: callee_node_types,
            call_names: inner_call_names,
            origin: None,
            capture_count: 0,
            baked_closures,
            higher_order_args: HashMap::new(),
            body: UnitBody::Real(callee_body),
        });

        specialized.insert(key, specialized_name);
    }

    for call in &found {
        let key = (call.callee_unit_name.clone(), call.erased.clone());
        let Some(specialized_name) = specialized.get(&key) else { continue };
        let positions: Vec<usize> = call.erased.iter().map(|(i, _)| *i).collect();
        let caller = &mut units[call.caller_idx];
        caller.call_names.insert(call.call_id, specialized_name.clone());
        caller.higher_order_args.insert(call.call_id, positions);
    }

    units.extend(new_units);
}

/// `(bare call name, concrete argument types, concrete return type) -> unit
/// name` — see the module's own doc comment for why this specific triple is
/// already a unique key, and which calls actually need it (only ones
/// dispatched directly to an already-concrete algebra impl).
type CallIndex = HashMap<(String, Vec<String>, String), String>;

fn build_call_index(units: &[ConcreteUnit]) -> CallIndex {
    let mut index = CallIndex::new();
    for u in units {
        // The bare method/fn name is whatever follows the last `::` in a
        // unit's own name — `"Ring::add<i32>"` -> `"add"`, `"fibonacci<i32>"`
        // -> `"fibonacci<i32>"` (no `::` at all, kept as-is; only ever
        // consulted for the algebra-impl case in practice, see below).
        let bare = u.name.rsplit_once("::").map(|(_, m)| m).unwrap_or(&u.name);
        let bare = bare.split('<').next().unwrap_or(bare).to_string();
        let arg_tys: Vec<String> = u.param_types.iter().map(Ty::to_string).collect();
        index.insert((bare, arg_tys, u.result.to_string()), u.name.clone());
    }
    index
}

// ---------------------------------------------------------------- CPS IR

pub type CVar = u32;

#[derive(Debug, Clone)]
pub enum CVal {
    Var(CVar),
    Int(u64),
    Float(f64),
    Bool(bool),
    Unit,
    /// A `CFunDef`'s own name, used as a call target.
    Label(String),
    /// A `let`-bound name currently bound to a lambda, not an ordinary
    /// runtime value — `captures` are this lambda's own free variables'
    /// `CVal`s, gathered from `env` *once*, at the `let` itself (snapshot-
    /// at-definition-time semantics, not by-reference — see `doc/backlog.md`'s
    /// own closure-conversion item for why that's the stated decision), in
    /// the same sorted-name order `lambda_free_vars` always produces. Which
    /// *unit* a given call actually resolves to isn't stored here at all —
    /// that's `ctx.call_names`' own job, resolved per call site exactly like
    /// any other generic instantiation (a `let`-bound lambda can itself be
    /// generic and get called at two different types, each needing its own
    /// resolution — see `mlir_lower.rs`'s "let polymorphism" test for the
    /// non-lambda analogue). Deliberately **never** written into a `LetPrim`/
    /// `App`'s own `args`/`func`, nor returned as an ordinary `k`-continuation
    /// value — it only ever lives inside `CEnv`, consumed exclusively by
    /// `convert_expr`'s own `Call`/`Path` arms, and is fully gone by the time
    /// a `CpsProgram` is built; `mlir_lower.rs` never sees this variant.
    Closure { captures: Vec<CVal> },
}

#[derive(Debug, Clone)]
pub enum PrimOp {
    /// `args = [base]`. `struct_ty` is the *base*'s own concrete type
    /// (`ctx.node_types` at the `FieldAccess` expression's own `base`
    /// node) — a struct value's MLIR representation carries no name of its
    /// own (`mlir_lower.rs` builds an anonymous `!llvm.struct<...>`), so
    /// `mlir_lower.rs` needs this to know *which* struct's own declared
    /// field order (hence `llvm.extractvalue`'s own required `position`)
    /// applies here — recovering it from the base's already-lowered MLIR
    /// `Value` alone isn't possible.
    Field { struct_ty: Ty, field: String },
    /// `args = [base, value]` — a real effect, mutating `base`'s own field
    /// in place (a struct is a stable reference, see `mlir_lower.rs::
    /// struct_llvm_type`'s own doc comment); its own bound result is unit
    /// and never read, same shape as `Store`'s own doc comment for arrays.
    /// `struct_ty`/`field` — see `Field`'s own doc comment for why the
    /// base's own concrete type has to be carried explicitly.
    FieldStore { struct_ty: Ty, field: String },
    Struct(String, Vec<String>),
    /// `args` are the element values, in order (`ExprKind::ArrayLit`).
    Array,
    /// `args = [value, count]` (`ExprKind::ArrayRepeat`).
    ArrayRepeat,
    /// `args = [array, index...]`. `array_ty` is the *array*'s own concrete
    /// type (`ctx.node_types` at `collect_index_chain`'s own returned base
    /// expression) — needed because an array's own MLIR representation
    /// depends on *where* it lives (a standalone array is a self-describing
    /// `memref`, but an array reached through a struct field is an opaque
    /// `!llvm.ptr` into that struct's own storage, carrying no shape of its
    /// own — see `mlir_lower.rs`'s own `struct_llvm_type`/`lower_array_load`
    /// doc comments); recovering the element type/dimensions from the
    /// already-lowered `Value` alone only works for the first case.
    Load { array_ty: Ty },
    /// `args = [array, index..., value]` — a real effect, mutating `array`
    /// in place; its own bound result is unit and never read. `array_ty` —
    /// see `Load`'s own doc comment.
    Store { array_ty: Ty },
    /// A call to a real, separately-compiled C-ABI symbol (`UnitBody::
    /// Extern`) — `param_types` is threaded through explicitly (from the
    /// callee's own `ConcreteUnit::param_types`, already on hand at
    /// `emit_call`'s call site) because MLIR lowering needs it twice: once
    /// to build the external declaration's own signature, and once to know
    /// each argument's *expected* MLIR type — a bare literal argument
    /// carries no width of its own in this IR, the same gap `LetPrim`'s own
    /// `ty` field exists to close for the result alone.
    Extern { symbol: String, param_types: Vec<Ty> },
    /// A reserved `mlir::dialect::op(...)` call (`ExprKind::Call` whose path
    /// starts with `"mlir"`, recognized structurally in `convert_expr` —
    /// see that match arm's own doc comment) — `op` is the real MLIR
    /// instruction name (`path.segments[1..].join(".")`, e.g. `arith.
    /// addi`), `attrs` is `ExprKind::Call::mlir_attrs` carried straight
    /// through unchanged (attribute name -> raw literal text, parsed via
    /// `Attribute::parse` only once lowering actually needs a real MLIR
    /// attribute value, not here). This is the *entire* hardcoded-in-Rust
    /// surface for primitive operations now -- no per-op-name Rust match
    /// left anywhere, matching `doc/hld.md`'s own "one generic 'emit this
    /// named MLIR op' primitive" goal.
    RawMlirOp { op: String, attrs: Vec<(String, String)> },
}

#[derive(Debug, Clone)]
pub enum CExpr {
    LetPrim { var: CVar, ty: Ty, op: PrimOp, args: Vec<CVal>, cont: Box<CExpr> },
    /// `func(args)` — for a *real* callee, `args`' own last entry is, by
    /// convention, the continuation to invoke with the result. A `PrimOp`
    /// never appears here — that's a `LetPrim`.
    App { func: CVal, args: Vec<CVal> },
    /// Introduces one or more local, single-use continuations — what a
    /// later, separate closure-conversion pass eventually flattens away.
    Fix { defs: Vec<CFunDef>, body: Box<CExpr> },
    /// Both arms are expected to end by tail-calling the same synthesized
    /// join continuation (see `convert_expr`'s own `ExprKind::If` arm) —
    /// `CExpr` itself doesn't enforce that, it's a convention of how this
    /// module builds one.
    If { cond: CVal, then_branch: Box<CExpr>, else_branch: Box<CExpr> },
}

#[derive(Debug, Clone)]
pub struct CFunDef {
    pub name: String,
    /// Ordinary parameters, plus — for a *real* top-level function
    /// specifically (see `CTopLevelFn`), one trailing continuation
    /// parameter. A `Fix`-local continuation (an `if`/loop join, or a real
    /// call's own resumption point) has no such trailing param of its own —
    /// it already *is* a continuation, nothing further to return to.
    pub params: Vec<CVar>,
    pub body: CExpr,
    /// `Some(one Ty per `params` entry, same order)` for a **loop's** own
    /// self-recursive `CFunDef` specifically — `None` everywhere else (a
    /// join, a real call's own resumption, or a top-level function, none of
    /// which need it: mlir_lower.rs already gets a join's own single value
    /// type from the enclosing function's `result_type`, and a real call's
    /// own resumption from the callee's signature — see `ExprKind::While`/
    /// `For`'s own doc comment for why a loop's own carried state needs
    /// this explicitly instead). A carried variable's own initial value
    /// (`gather_carried`) can be a bare, width-less literal CVal (`total =
    /// 0.0`, never wrapped in a `LetPrim`) — with nothing else in the CPS
    /// IR recording its real type, `mlir_lower.rs::lower_loop` used to guess
    /// via the *enclosing function's* own `result_type`, wrong whenever a
    /// carried value's type differs from it (found by direct testing: an
    /// `f32`-accumulating loop inside an `f32`-returning function masked
    /// this for every loop test until one carried an `i32` index *and* an
    /// `f32` accumulator at once — the index got materialized as `f32` by
    /// mistake, corrupting the generated MLIR).
    pub carried_types: Option<Vec<Ty>>,
}

/// A top-level `CFunDef` specifically — everything in `CpsProgram::funcs`,
/// as opposed to a `Fix`-local one (an `if`/loop join, or a real call's own
/// resumption point), which stays a bare `CFunDef` with no cleave-level
/// typing of its own. Carries the extra metadata a later MLIR-lowering pass
/// needs to build a real `func.func` signature — the CPS IR itself is
/// otherwise untyped past each `LetPrim`'s own `ty`.
pub struct CTopLevelFn {
    pub def: CFunDef,
    /// `def.params`' own cleave types, same length and order (excludes
    /// `k_ret` below, which has no cleave-level type of its own).
    pub param_types: Vec<Ty>,
    /// This function's own cleave-level return type.
    pub result: Ty,
    /// `def.params`' own trailing "return continuation" parameter — tail-
    /// calling it (`App { func: Var(k_ret), args: [v] }`) is exactly this
    /// function's own `return v`, not a real call. Kept as an explicit
    /// field (not re-derived from "params' own last entry") so a consumer
    /// like a later MLIR-lowering pass can recognize the pattern
    /// unambiguously rather than relying on positional convention.
    pub k_ret: CVar,
    /// Threaded straight through from `ConcreteUnit::origin` — see its own
    /// doc comment. `ConcreteUnit` itself doesn't survive past `convert_
    /// program` (it's consumed converting each unit into a `CTopLevelFn`),
    /// so anything downstream of *this* IR that still needs a unit's own
    /// algebra origin (an e-graph pass matching an `axiom`'s declared
    /// algebra/method against a real call site, say) needs it duplicated
    /// here rather than parsed back out of `def.name`.
    pub origin: Option<(String, String)>,
}

pub struct CpsProgram {
    pub funcs: Vec<CTopLevelFn>,
}

// ---------------------------------------------------------------- conversion

/// `pub(crate)`, not private — reused as-is by a later e-graph pass
/// (`egraph.rs`) to mint fresh `CVar`s/labels while reconstructing a
/// rewritten segment back into CPS form, the identical need this module's
/// own conversion already has.
pub(crate) struct FreshVars {
    next_var: Cell<u32>,
    next_label: Cell<u32>,
}

impl FreshVars {
    pub(crate) fn new() -> Self {
        FreshVars { next_var: Cell::new(0), next_label: Cell::new(0) }
    }

    pub(crate) fn starting_at(next_var: CVar) -> Self {
        FreshVars { next_var: Cell::new(next_var), next_label: Cell::new(0) }
    }

    pub(crate) fn var(&self) -> CVar {
        let v = self.next_var.get();
        self.next_var.set(v + 1);
        v
    }

    pub(crate) fn label(&self, hint: &str) -> String {
        let n = self.next_label.get();
        self.next_label.set(n + 1);
        format!("{hint}${n}")
    }
}

struct Ctx<'a> {
    units: &'a HashMap<String, ConcreteUnit>,
    call_index: &'a CallIndex,
    node_types: &'a HashMap<NodeId, Ty>,
    call_names: &'a HashMap<NodeId, String>,
    /// Stage B only — see `ConcreteUnit::higher_order_args`'s own doc
    /// comment.
    higher_order_args: &'a HashMap<NodeId, Vec<usize>>,
    fresh: &'a FreshVars,
}

type CEnv = HashMap<String, CVal>;

/// Converts every `ConcreteUnit` with a real body into one `CFunDef` —
/// `extern` units never get one at all (see `PrimOp::Extern`, produced
/// directly at their own call sites instead).
pub fn convert_program(units: Vec<ConcreteUnit>) -> CpsProgram {
    let call_index = build_call_index(&units);
    let by_name: HashMap<String, ConcreteUnit> = units.into_iter().map(|u| (u.name.clone(), u)).collect();
    let fresh = FreshVars::new();
    let mut funcs = Vec::new();
    // Sorted, not raw `by_name.values()` -- `HashMap` iteration order isn't
    // just unstable across runs (`std`'s randomized per-process hasher
    // seed), it directly drives fresh var/label *numbering* here (assigned
    // as each unit is visited, shared across every unit via one `fresh`),
    // making otherwise-identical output differ run to run. Sorting first
    // makes CPS conversion fully deterministic and reproducible -- the same
    // property `funcs.sort_by` below already gives the *display* order,
    // just applied before numbering happens instead of only after.
    let mut sorted_units: Vec<&ConcreteUnit> = by_name.values().collect();
    sorted_units.sort_by(|a, b| a.name.cmp(&b.name));
    for unit in sorted_units {
        let UnitBody::Real(body) = &unit.body else { continue };
        // A unit with a `Ty::Fn`-typed parameter of its own can never be
        // converted/called *as declared* — a lambda has no runtime
        // representation at all (see `CVal::Closure`'s own doc comment), so
        // every real call to it necessarily goes through Stage B's own
        // per-callable redirection instead (`build_higher_order_
        // specializations`), which erases that very parameter from its own
        // specialized copy's signature. The *original*, unspecialized unit
        // stays in `units` regardless (nothing removes it — harmless, and
        // simpler than trying to prove no caller needs it) but converting
        // its own body directly would immediately panic on the erased
        // parameter's own now-unresolvable call (`resolve_call` finds
        // nothing, since only a Stage-B-produced specialization ever gets a
        // `call_names` override for it) — skipped here, not emitted at all,
        // since it's genuinely unreachable by construction: nothing in this
        // language can produce a runtime value of function type to call it
        // with in the first place.
        if unit.param_types.iter().any(|t| matches!(t, Ty::Fn(..))) {
            continue;
        }
        let ctx = Ctx {
            units: &by_name,
            call_index: &call_index,
            node_types: &unit.node_types,
            call_names: &unit.call_names,
            higher_order_args: &unit.higher_order_args,
            fresh: &fresh,
        };
        let mut env = CEnv::new();
        let mut params = Vec::with_capacity(unit.params.len() + 1);
        for p in &unit.params {
            let v = fresh.var();
            env.insert(p.name.clone(), CVal::Var(v));
            params.push(v);
        }
        // Stage B only: an erased, higher-order parameter this unit's own
        // signature no longer declares at all (see `ConcreteUnit::baked_
        // closures`'s own doc comment) still needs a name bound in `env` --
        // to a `CVal::Closure` over whichever of the ordinary params just
        // bound above are its own resolved callable's leading captures, so
        // `convert_expr`'s own existing (Stage A) `Call`-arm `CVal::Closure`
        // handling picks it up exactly like any directly `let`-bound lambda.
        for (erased_name, capture_param_names) in &unit.baked_closures {
            let captures: Vec<CVal> = capture_param_names
                .iter()
                .map(|n| env.get(n).cloned().unwrap_or_else(|| panic!("CPS: baked closure `{erased_name}`'s own capture param `{n}` unexpectedly unbound")))
                .collect();
            env.insert(erased_name.clone(), CVal::Closure { captures });
        }
        let k_ret = fresh.var();
        params.push(k_ret);
        let cexpr = convert_block(body, &env, &ctx, &|v, _env| CExpr::App { func: CVal::Var(k_ret), args: vec![v] });
        funcs.push(CTopLevelFn {
            def: CFunDef { name: unit.name.clone(), params, body: cexpr, carried_types: None },
            param_types: unit.param_types.clone(),
            result: unit.result.clone(),
            k_ret,
            origin: unit.origin.clone(),
        });
    }
    // Deterministic output order — `HashMap` iteration isn't stable.
    funcs.sort_by(|a, b| a.def.name.cmp(&b.def.name));
    CpsProgram { funcs }
}

fn convert_block(block: &Block, env: &CEnv, ctx: &Ctx, k: &dyn Fn(CVal, &CEnv) -> CExpr) -> CExpr {
    convert_stmts(&block.stmts, env.clone(), ctx, &|env| match &block.tail {
        Some(tail) => convert_expr(tail, &env, ctx, k),
        None => k(CVal::Unit, &env),
    })
}

fn convert_stmts(stmts: &[Stmt], env: CEnv, ctx: &Ctx, k: &dyn Fn(CEnv) -> CExpr) -> CExpr {
    let Some((stmt, rest)) = stmts.split_first() else {
        return k(env);
    };
    match &stmt.kind {
        // `mutable` doesn't matter to conversion itself -- both an ordinary
        // `let` and a `let mut` just (re)bind a name in `env`; the ML-style
        // generalize-vs-monomorphic distinction it drives is `infer.rs`'s
        // own concern, already resolved by the time a fully concrete
        // `ConcreteUnit` reaches this module.
        // A lambda has no runtime `CVal` of its own -- `convert_expr` never
        // handles `ExprKind::Lambda` (still an unconditional panic there,
        // see its own catch-all — that's deliberate: a bare lambda literal
        // used anywhere *except* directly `let`-bound, e.g. passed straight
        // as a call argument with no prior `let`, is explicitly out of
        // scope for this pass, matching `doc/backlog.md`'s own closure-
        // conversion item). Intercepted here instead: `name` gets bound to
        // a `CVal::Closure`, snapshotting each of the lambda's own free
        // variables' *current* `CVal` from `env` right now (see `CVal::
        // Closure`'s own doc comment for why that's a snapshot, not a
        // by-reference capture) — which concrete unit a later call to
        // `name(...)` actually resolves to is deferred to that call site
        // itself (`ctx.call_names`, exactly like any other generic
        // instantiation), not decided here.
        StmtKind::Let { name, value, .. } if matches!(value.kind, ExprKind::Lambda { .. }) => {
            let ExprKind::Lambda { params, body, .. } = &value.kind else { unreachable!() };
            let free = lambda_free_vars(params, body, ctx.node_types);
            let captures: Vec<CVal> = sorted_capture_names(&free)
                .into_iter()
                .map(|n| {
                    env.get(&n)
                        .cloned()
                        .unwrap_or_else(|| panic!("CPS: lambda's own captured variable `{n}` unexpectedly unbound at its own `let`"))
                })
                .collect();
            let mut env2 = env.clone();
            env2.insert(name.clone(), CVal::Closure { captures });
            convert_stmts(rest, env2, ctx, k)
        }
        StmtKind::Let { name, value, .. } => {
            let name = name.clone();
            convert_expr(value, &env, ctx, &|v, env| {
                let mut env2 = env.clone();
                env2.insert(name.clone(), v);
                convert_stmts(rest, env2, ctx, k)
            })
        }
        StmtKind::Expr(e) => convert_expr(e, &env, ctx, &|_v, env| convert_stmts(rest, env.clone(), ctx, k)),
        StmtKind::Assign { target, value } => match &target.kind {
            ExprKind::Path(p) => {
                let name = p.segments.join("::");
                convert_expr(value, &env, ctx, &|v, env| {
                    let mut env2 = env.clone();
                    env2.insert(name.clone(), v);
                    convert_stmts(rest, env2, ctx, k)
                })
            }
            // A chain of one or more `Index` *groups* (`a[i,j]`'s own single
            // bracket group, or two separate bracket pairs `a[i][j]` — both
            // still land here, and both still collapse into *one* combined,
            // multi-index `Store` via `collect_index_chain`, which now walks
            // group-by-group rather than index-by-index) — see the module's
            // own "Arrays" doc comment for why this matters for correctness,
            // not just efficiency: a write through an *intermediate* single-
            // index `Load` (getting "the row", then storing into that) would
            // only be correct if `Load` on an array-of-arrays element
            // aliased the original storage rather than copying it out — a
            // real, load-bearing representation choice this module never
            // actually makes, so this instead never produces that
            // intermediate `Load` at all: the whole chain resolves to one
            // flat offset in a single effect. `infer.rs`'s own `StmtKind::
            // Assign` guard already rejects a non-array base before this is
            // ever reached, so this is always a real array, never an
            // `Index`-algebra target.
            ExprKind::Index(..) => {
                let (array_expr, index_exprs) = collect_index_chain(target);
                let array_ty = ctx.node_types[&array_expr.id].clone();
                convert_expr(array_expr, &env, ctx, &|array_val, env| {
                    convert_expr_list(&index_exprs, env, ctx, &|index_vals, env| {
                        convert_expr(value, env, ctx, &|new_val, env| {
                            let var = ctx.fresh.var();
                            let mut args = vec![array_val.clone()];
                            args.extend(index_vals.clone());
                            args.push(new_val);
                            CExpr::LetPrim {
                                var,
                                ty: Ty::Con("()".to_string()),
                                op: PrimOp::Store { array_ty: array_ty.clone() },
                                args,
                                cont: Box::new(convert_stmts(rest, env.clone(), ctx, k)),
                            }
                        })
                    })
                })
            }
            // A struct is a stable reference mutated in place, same as an
            // array (see the module's own "Arrays" doc comment) — a field
            // write is a real effect through the base's own existing
            // pointer, not a functional update, so this needs no join/
            // carried-state threading at all, mirroring `Index`'s own arm
            // just above.
            ExprKind::FieldAccess(base, field) => {
                let struct_ty = ctx.node_types[&base.id].clone();
                convert_expr(base, &env, ctx, &|base_val, env| {
                    convert_expr(value, env, ctx, &|new_val, env| {
                        let var = ctx.fresh.var();
                        CExpr::LetPrim {
                            var,
                            ty: Ty::Con("()".to_string()),
                            op: PrimOp::FieldStore { struct_ty: struct_ty.clone(), field: field.clone() },
                            args: vec![base_val.clone(), new_val],
                            cont: Box::new(convert_stmts(rest, env.clone(), ctx, k)),
                        }
                    })
                })
            }
            other => panic!("CPS: unexpected assignment target {other:?}"),
        },
    }
}

fn convert_expr(expr: &Expr, env: &CEnv, ctx: &Ctx, k: &dyn Fn(CVal, &CEnv) -> CExpr) -> CExpr {
    match &expr.kind {
        // A plain numeric literal widened to `Complex<T>` (`4 + 2i`, `4`'s
        // own side — see `infer.rs`'s `check_pending_constraints`'s own
        // `Complex`-widening special case) — real = the literal's own
        // value, imag = `0`, both materialized at `T`'s own concrete width.
        ExprKind::NumberLit { text, .. } if matches!(&ctx.node_types[&expr.id], Ty::App(name, _) if name == "Complex") => {
            complex_literal(ctx, expr, text, "0", env, k)
        }
        ExprKind::NumberLit { text, .. } => {
            let ty = &ctx.node_types[&expr.id];
            k(parse_number(text, ty), env)
        }
        // `doc/backlog.md`'s own "Complex literals" item — `4i`'s own
        // resolved type is always `Complex<T>` (`infer.rs`'s own
        // `ImaginaryLit` handling never defaults anywhere else) — real =
        // `0`, imag = the literal's own value.
        ExprKind::ImaginaryLit { text, .. } => complex_literal(ctx, expr, "0", text, env, k),
        ExprKind::BoolLit(b) => k(CVal::Bool(*b), env),
        ExprKind::Path(p) => {
            let name = p.segments.join("::");
            let v = match env.get(&name) {
                // `name` refers to a lambda, used here as an ordinary value
                // (aliased to another name, passed as an argument, returned,
                // stored in a field/array, ...) rather than called directly
                // by name — none of those are implemented yet (see `CVal::
                // Closure`'s own doc comment; a lambda *returned* from a
                // function or stored in a struct/array field is explicitly
                // out of scope for this pass per `doc/backlog.md`'s own
                // closure-conversion item) — a clean, explicit panic here
                // rather than silently letting a `Closure` leak into an
                // ordinary `LetPrim`/`App` argument position, which
                // `mlir_lower.rs` has no representation for at all.
                Some(CVal::Closure { .. }) => {
                    panic!("CPS: `{name}` names a lambda used as an ordinary value (not called directly by name) -- not supported yet")
                }
                Some(v) => v.clone(),
                // A const generic referenced as an ordinary value (`[v; N]`,
                // `for i in 0..N`) is never bound as a real parameter/`let`
                // -- by the time a `ConcreteUnit` reaches this module,
                // monomorphization has already substituted its own
                // `node_types` entry for this exact reference down to a
                // resolved `Ty::Const` (see `infer.rs`'s own
                // `seed_const_generics`/`const_widths`), so it converts the
                // same way a literal does rather than needing to be seeded
                // into `env` at all.
                None => match &ctx.node_types[&expr.id] {
                    Ty::Const(ConstValue::Int(n)) => CVal::Int(*n),
                    Ty::Const(ConstValue::Bool(b)) => CVal::Bool(*b),
                    _ => panic!("CPS: unbound variable `{name}`"),
                },
            };
            k(v, env)
        }
        ExprKind::FieldAccess(base, field) => {
            let struct_ty = ctx.node_types[&base.id].clone();
            convert_expr(base, env, ctx, &|base_val, env| {
                let var = ctx.fresh.var();
                CExpr::LetPrim {
                    var,
                    ty: ctx.node_types[&expr.id].clone(),
                    op: PrimOp::Field { struct_ty: struct_ty.clone(), field: field.clone() },
                    args: vec![base_val],
                    cont: Box::new(k(CVal::Var(var), env)),
                }
            })
        }
        ExprKind::StructLit(path, _, fields) => {
            let struct_name = path.segments.join("::");
            let field_names: Vec<String> = fields.iter().map(|(n, _)| n.clone()).collect();
            let values: Vec<&Expr> = fields.iter().map(|(_, v)| v).collect();
            convert_expr_list(&values, env, ctx, &|arg_vals, env| {
                let var = ctx.fresh.var();
                CExpr::LetPrim {
                    var,
                    ty: ctx.node_types[&expr.id].clone(),
                    op: PrimOp::Struct(struct_name.clone(), field_names.clone()),
                    args: arg_vals,
                    cont: Box::new(k(CVal::Var(var), env)),
                }
            })
        }
        // A reserved raw-MLIR-op call (`mlir::arith::addi(a, b)`) -- checked
        // structurally, *before* any `resolve_call`/`emit_call` lookup, the
        // same way `infer.rs::infer_call` short-circuits it during type
        // checking. Straight-line like `PrimOp::Extern`: an MLIR op is
        // always synchronous, no `Fix`/continuation needed just because its
        // meaning happens to live outside the algebra registry.
        ExprKind::Call(path, _, args, mlir_attrs) if path.segments.first().map(String::as_str) == Some("mlir") => {
            let op = path.segments[1..].join(".");
            let arg_refs: Vec<&Expr> = args.iter().collect();
            let result_ty = ctx.node_types[&expr.id].clone();
            let attrs = mlir_attrs.clone();
            convert_expr_list(&arg_refs, env, ctx, &move |arg_vals, env| {
                let var = ctx.fresh.var();
                CExpr::LetPrim {
                    var,
                    ty: result_ty.clone(),
                    op: PrimOp::RawMlirOp { op: op.clone(), attrs: attrs.clone() },
                    args: arg_vals,
                    cont: Box::new(k(CVal::Var(var), env)),
                }
            })
        }
        ExprKind::Call(path, _, args, ..) => {
            let name = path.segments.join("::");
            let arg_refs: Vec<&Expr> = args.iter().collect();
            let arg_ids: Vec<NodeId> = args.iter().map(|a| a.id).collect();
            let result_ty = ctx.node_types[&expr.id].clone();
            // Checked *first*, before falling through to `resolve_call`'s
            // own tiers — a local lambda binding shadowing a same-named
            // top-level `fn` must resolve to the lambda (`let f = fn(x){x};
            // f(5)` even when a top-level `fn f` also happens to exist) —
            // matches `monomorphize.rs`'s own `collect_instantiations_expr`,
            // which already gives its own scope-tracking the identical
            // priority for the identical reason.
            if let Some(CVal::Closure { captures }) = env.get(&name) {
                let captures = captures.clone();
                return convert_expr_list(&arg_refs, env, ctx, &|arg_vals, env| {
                    // Resolved per *this* call site, exactly like any other
                    // generic instantiation (`ctx.call_names`, built by
                    // `monomorphize.rs`'s own lambda worklist) — never a
                    // single fixed unit for the `let`-bound name as a whole,
                    // since the same lambda can be called at two different
                    // concrete types from two different call sites (let-
                    // polymorphism, mirroring an ordinary generic `fn`).
                    let mangled = ctx.call_names.get(&expr.id).unwrap_or_else(|| {
                        panic!(
                            "CPS: call to lambda-bound `{name}` at a call site monomorphize.rs never resolved a specialization for"
                        )
                    });
                    let callee = ctx
                        .units
                        .get_key_value(mangled.as_str())
                        .map(|(k, _)| k.as_str())
                        .unwrap_or_else(|| panic!("CPS: lambda call resolved to `{mangled}`, but no such unit exists"));
                    let mut full_args = captures.clone();
                    full_args.extend(arg_vals);
                    emit_call(callee, full_args, result_ty.clone(), ctx, env, k)
                });
            }
            // Stage B: one or more of *this* call's own arguments name a
            // callable (lambda-bound, or a bare top-level `fn`) that `collect_
            // units`'s own `build_higher_order_specializations` already baked
            // directly into a specialized callee unit (`ctx.call_names[&expr.
            // id]`, overridden to that specialized unit's own name — `resolve_
            // call`'s ordinary first tier already picks it up unchanged, no
            // special resolution needed here). What *does* need special
            // handling: an erased argument doesn't convert to an ordinary
            // `CVal` at all (my own `ExprKind::Path` arm above would panic on
            // it, on purpose — see its own doc comment) — its own captures get
            // gathered from `env` and spliced in at that position instead,
            // exactly the same snapshot mechanism as calling a lambda
            // directly (Stage A), just landing in the *middle* of an ordinary
            // argument list here rather than always at the front.
            if let Some(erased) = ctx.higher_order_args.get(&expr.id) {
                let mut prelude: Vec<CVal> = Vec::new();
                for &idx in erased {
                    let callable_name = match &args[idx].kind {
                        ExprKind::Path(p) => p.segments.join("::"),
                        other => panic!("CPS: higher-order argument {idx} of `{name}` must be a bare name, found {other:?}"),
                    };
                    if let Some(CVal::Closure { captures }) = env.get(&callable_name) {
                        prelude.extend(captures.iter().cloned());
                    }
                    // Else: a bare top-level `fn` name, with no captures of
                    // its own — nothing to splice in for this position.
                }
                let remaining: Vec<&Expr> = args.iter().enumerate().filter(|(i, _)| !erased.contains(i)).map(|(_, a)| a).collect();
                return convert_expr_list(&remaining, env, ctx, &move |arg_vals, env| {
                    let callee = resolve_call(&name, expr.id, &arg_ids, ctx);
                    let mut full_args = prelude.clone();
                    full_args.extend(arg_vals);
                    emit_call(callee, full_args, result_ty.clone(), ctx, env, k)
                });
            }
            convert_expr_list(&arg_refs, env, ctx, &|arg_vals, env| {
                let callee = resolve_call(&name, expr.id, &arg_ids, ctx);
                // A *self*-recursive call to a captured lambda's own unit,
                // from inside that same unit's own body, is the only way a
                // call resolving to a `<lambda#...>` unit ever reaches this
                // plain fallback instead of the `env.get(&name) == Closure`
                // fast path above (Stage A) — an ordinary reference to a
                // captured lambda is always caught there first, via a real
                // `CVal::Closure` bound by its own enclosing `let`. A self-
                // reference inside the lambda's own separately-converted
                // unit never gets that binding (its own unit starts a fresh
                // `env`) — but it needs the *same* captures this very unit
                // was itself called with, already sitting in `env` under
                // their own original names (this unit's own leading
                // `capture_count` params — see `convert_program`'s setup).
                let mut full_args = Vec::new();
                if let Some(unit) = ctx.units.get(callee) {
                    for p in &unit.params[..unit.capture_count] {
                        full_args.push(env.get(&p.name).cloned().unwrap_or_else(|| {
                            panic!("CPS: self-recursive call to `{name}` needs its own capture `{}`, unexpectedly unbound", p.name)
                        }));
                    }
                }
                full_args.extend(arg_vals);
                emit_call(callee, full_args, result_ty.clone(), ctx, env, k)
            })
        }
        ExprKind::If { cond, then_branch, else_branch } => convert_expr(cond, env, ctx, &|cond_val, env| {
            // Both arms tail-call the same synthesized join continuation
            // rather than each inlining `k` directly — otherwise "what
            // happens after the if" would be duplicated into every branch
            // (and, once nested, blow up combinatorially). `k` itself is
            // invoked exactly once, in the join continuation's own body.
            // Beyond the if's own value, the join also carries whatever
            // enclosing-scope names either arm might reassign (see the
            // module's own "Mutation across control flow" doc comment) —
            // computed once, statically, from the source AST, not from
            // `env` itself.
            let mut mutated: HashMap<String, Ty> = mutated_free_vars(then_branch, &HashSet::new(), ctx);
            match else_branch {
                Some(eb) => match &**eb {
                    ElseBranch::If(e) => mutated.extend(mutated_free_vars_expr(e, &HashSet::new(), ctx)),
                    ElseBranch::Block(b) => mutated.extend(mutated_free_vars(b, &HashSet::new(), ctx)),
                },
                None => {}
            }
            let mut names: Vec<String> = mutated.keys().cloned().collect();
            names.sort();
            names.dedup();
            let carried: Vec<(String, CVar)> = names.iter().map(|n| (n.clone(), ctx.fresh.var())).collect();
            let carried_types: Vec<Ty> = names.iter().map(|n| mutated[n].clone()).collect();

            let result_var = ctx.fresh.var();
            let join_label = ctx.fresh.label("j");

            let then_cexpr =
                convert_block(then_branch, env, ctx, &|v, branch_env| tail_call_join(branch_env, v, &carried, &join_label));
            let else_cexpr = match else_branch {
                Some(eb) => match &**eb {
                    ElseBranch::If(e) => convert_expr(e, env, ctx, &|v, branch_env| tail_call_join(branch_env, v, &carried, &join_label)),
                    ElseBranch::Block(b) => {
                        convert_block(b, env, ctx, &|v, branch_env| tail_call_join(branch_env, v, &carried, &join_label))
                    }
                },
                None => tail_call_join(env, CVal::Unit, &carried, &join_label),
            };

            let mut join_params = vec![result_var];
            join_params.extend(carried.iter().map(|(_, v)| *v));
            let join_body = {
                let mut env2 = env.clone();
                for (name, var) in &carried {
                    env2.insert(name.clone(), CVal::Var(*var));
                }
                k(CVal::Var(result_var), &env2)
            };

            // `join.carried_types`'s own order matches `join_params`
            // exactly: the `if`'s own value type first (`ctx.node_types` at
            // this expression's own node), then one `Ty` per carried outer
            // variable, in the same order `carried` itself was built —
            // `mlir_lower.rs::lower_if` needs all of them explicitly, the
            // same reason a loop's own `CFunDef` does (see `CFunDef::
            // carried_types`'s own doc comment).
            let mut join_types = vec![ctx.node_types[&expr.id].clone()];
            join_types.extend(carried_types);

            CExpr::Fix {
                defs: vec![CFunDef { name: join_label, params: join_params, body: join_body, carried_types: Some(join_types) }],
                body: Box::new(CExpr::If { cond: cond_val, then_branch: Box::new(then_cexpr), else_branch: Box::new(else_cexpr) }),
            }
        }),
        ExprKind::While { cond, body } => {
            // Loop-carried state: whatever enclosing-scope names `cond` or
            // `body` might reassign -- `cond` is included too (not just
            // `body`), since it's re-evaluated fresh every iteration and a
            // mutation happening *during* that re-evaluation must still
            // persist into the next one. Both outcomes (recurse, or hand
            // off to `k`) are genuine tail calls, so — unlike `ExprKind::If`
            // above, used as an ordinary sub-expression needing to feed a
            // shared join — no separate join continuation is needed here:
            // each arm already terminates on its own, `k` is only ever
            // reached via the exit path.
            let mut mutated: HashMap<String, Ty> = mutated_free_vars_expr(cond, &HashSet::new(), ctx);
            mutated.extend(mutated_free_vars(body, &HashSet::new(), ctx));
            let mut names: Vec<String> = mutated.keys().cloned().collect();
            names.sort();
            names.dedup();
            let carried: Vec<(String, CVar)> = names.iter().map(|n| (n.clone(), ctx.fresh.var())).collect();
            let carried_types: Vec<Ty> = names.iter().map(|n| mutated[n].clone()).collect();

            let loop_label = ctx.fresh.label("loop");
            let mut loop_env = env.clone();
            for (name, var) in &carried {
                loop_env.insert(name.clone(), CVal::Var(*var));
            }

            let check = convert_expr(cond, &loop_env, ctx, &|cond_val, cond_env| {
                let then_cexpr = convert_block(body, cond_env, ctx, &|_v, body_env| CExpr::App {
                    func: CVal::Label(loop_label.clone()),
                    args: gather_carried(&carried, body_env),
                });
                let else_cexpr = k(CVal::Unit, cond_env);
                CExpr::If { cond: cond_val, then_branch: Box::new(then_cexpr), else_branch: Box::new(else_cexpr) }
            });

            let params: Vec<CVar> = carried.iter().map(|(_, v)| *v).collect();
            let init_args = gather_carried(&carried, env);
            CExpr::Fix {
                defs: vec![CFunDef { name: loop_label.clone(), params, body: check, carried_types: Some(carried_types) }],
                body: Box::new(CExpr::App { func: CVal::Label(loop_label), args: init_args }),
            }
        }
        ExprKind::For { var, start, end, body } => convert_expr(start, env, ctx, &|start_val, env| {
            convert_expr(end, env, ctx, &|end_val, env| {
                // The loop's own index is threaded exactly like `for`'s own
                // one intrinsic piece of carried state always was (see
                // Stage 3); on top of it, whatever enclosing-scope names
                // `body` reassigns get carried the same way `while`'s own
                // do. `var` itself seeds the shadow set: it's never a
                // `let mut`, so an (illegal) assignment to it wouldn't have
                // anywhere outer to escape to regardless.
                let mut shadowed = HashSet::new();
                shadowed.insert(var.clone());
                let mutated: HashMap<String, Ty> = mutated_free_vars(body, &shadowed, ctx);
                let mut names: Vec<String> = mutated.keys().cloned().collect();
                names.sort();
                let carried: Vec<(String, CVar)> = names.iter().map(|n| (n.clone(), ctx.fresh.var())).collect();
                let carried_types: Vec<Ty> = names.iter().map(|n| mutated[n].clone()).collect();

                // `lt`/`add` are never written as calls in the source (the
                // grammar bakes the count-up directly into `for`), so
                // they're resolved the same way an operator desugars to a
                // `Call` elsewhere, just without a real `Expr` call node to
                // hang the lookup off of -- straight through
                // `ctx.call_index` (see `resolve_call`'s own doc comment,
                // tier 3).
                // A `for` loop's own counter type is deliberately "Int,
                // unconstrained width" (see `infer.rs`'s own `ExprKind::For`
                // comment) -- when nothing else pins it more specifically,
                // it deliberately *stays* an unresolved type variable at
                // declaration time, specifically so it doesn't conflict with
                // a const generic's own later resolution if the loop bound
                // happens to *name* one (`for i in 0..N`; see
                // `apply_defaults`'s own `const_widths` guard). Monomorphization
                // then resolves that variable to `N`'s own concrete *value*
                // (`Ty::Const`), not a width -- not usable here at all
                // (`ConstValue::Int` never carried a width tag to recover in
                // the first place). Falls back to the same `i32`
                // `apply_defaults` itself would have chosen had the guard not
                // deferred it: if the counter genuinely needed a *different*
                // width, some other real use of it (a comparison/arithmetic
                // op against an already-concrete value elsewhere in the
                // body) would already have pinned an ordinary `Ty::Con` here
                // directly, never going through `Ty::Const` at all.
                let idx_ty = match &ctx.node_types[&start.id] {
                    Ty::Const(_) => Ty::Con("i32".to_string()),
                    other => other.clone(),
                };
                let bool_ty = Ty::Con("bool".to_string());
                let lt_unit = resolve_synthetic_binop("lt", &idx_ty, &bool_ty, ctx).to_string();
                let add_unit = resolve_synthetic_binop("add", &idx_ty, &idx_ty, ctx).to_string();
                let i_var = ctx.fresh.var();
                let loop_label = ctx.fresh.label("loop");

                let mut body_env = env.clone();
                body_env.insert(var.clone(), CVal::Var(i_var));
                for (name, v) in &carried {
                    body_env.insert(name.clone(), CVal::Var(*v));
                }

                let cond_check = emit_call(&lt_unit, vec![CVal::Var(i_var), end_val], bool_ty, ctx, &body_env, &|cond_val, cond_env| {
                    let then_cexpr = convert_block(body, cond_env, ctx, &|_v, body_end_env| {
                        emit_call(&add_unit, vec![CVal::Var(i_var), CVal::Int(1)], idx_ty.clone(), ctx, body_end_env, &|next_i, incr_env| {
                            let mut args = vec![next_i];
                            args.extend(gather_carried(&carried, incr_env));
                            CExpr::App { func: CVal::Label(loop_label.clone()), args }
                        })
                    });
                    let else_cexpr = k(CVal::Unit, cond_env);
                    CExpr::If { cond: cond_val, then_branch: Box::new(then_cexpr), else_branch: Box::new(else_cexpr) }
                });

                let mut params = vec![i_var];
                params.extend(carried.iter().map(|(_, v)| *v));
                let mut init_args = vec![start_val.clone()];
                init_args.extend(gather_carried(&carried, env));
                let mut all_carried_types = vec![idx_ty.clone()];
                all_carried_types.extend(carried_types);

                CExpr::Fix {
                    defs: vec![CFunDef { name: loop_label.clone(), params, body: cond_check, carried_types: Some(all_carried_types) }],
                    body: Box::new(CExpr::App { func: CVal::Label(loop_label), args: init_args }),
                }
            })
        }),
        // `doc/backlog-done.md`'s own "`for x in array`" item — element-based
        // iteration over a real, homogeneous array. Reuses `ExprKind::For`'s
        // own `Fix`-building shape almost verbatim right above (`mutated_
        // free_vars`, the `lt`/`add` synthetic-binop resolution, the carried-
        // vars machinery) rather than reinventing it — `start`/`end` are
        // always `0`/the array's own known size (`ctx.node_types[&iter.id]`,
        // guaranteed a concrete `Ty::Array(_, Ty::Const(_))` here, monomorphization
        // has already run), and `idx_ty` is always plain `i32` (no user-written
        // bound expression to recover a different width from). The one real
        // difference: `body_env` gets one extra `LetPrim{Load}` binding `var`
        // to the *loaded element*, not the index — exactly the way
        // `ExprKind::Index`'s own conversion just below already builds a
        // `Load` — so `body` (the user's own AST, referencing `var`) needs no
        // rewriting at all, it just resolves `var` through the environment
        // like any other local, same as `var` already resolving to the
        // *index* CVar in the range-based `For` arm right above.
        ExprKind::ForIn { var, iter, body } => convert_expr(iter, env, ctx, &|iter_val, env| {
            let array_ty = ctx.node_types[&iter.id].clone();
            let (elem_ty, n) = match &array_ty {
                Ty::Array(elem, size) => match size.as_ref() {
                    Ty::Const(ConstValue::Int(n)) => ((**elem).clone(), *n),
                    other => panic!("CPS: for-in over an array with a non-concrete size {other:?} -- should have been resolved by monomorphization"),
                },
                other => panic!("CPS: for-in over a non-array type {other:?} -- infer.rs should have rejected this already"),
            };

            let mut shadowed = HashSet::new();
            shadowed.insert(var.clone());
            let mutated: HashMap<String, Ty> = mutated_free_vars(body, &shadowed, ctx);
            let mut names: Vec<String> = mutated.keys().cloned().collect();
            names.sort();
            let carried: Vec<(String, CVar)> = names.iter().map(|n| (n.clone(), ctx.fresh.var())).collect();
            let carried_types: Vec<Ty> = names.iter().map(|n| mutated[n].clone()).collect();

            let idx_ty = Ty::Con("i32".to_string());
            let bool_ty = Ty::Con("bool".to_string());
            let lt_unit = resolve_synthetic_binop("lt", &idx_ty, &bool_ty, ctx).to_string();
            let add_unit = resolve_synthetic_binop("add", &idx_ty, &idx_ty, ctx).to_string();
            let i_var = ctx.fresh.var();
            let loop_label = ctx.fresh.label("loop");

            let mut body_env = env.clone();
            for (name, v) in &carried {
                body_env.insert(name.clone(), CVal::Var(*v));
            }

            let cond_check = emit_call(&lt_unit, vec![CVal::Var(i_var), CVal::Int(n)], bool_ty, ctx, &body_env, &|cond_val, cond_env| {
                let elem_var = ctx.fresh.var();
                let mut elem_env = cond_env.clone();
                elem_env.insert(var.clone(), CVal::Var(elem_var));
                let then_cexpr = CExpr::LetPrim {
                    var: elem_var,
                    ty: elem_ty.clone(),
                    op: PrimOp::Load { array_ty: array_ty.clone() },
                    args: vec![iter_val.clone(), CVal::Var(i_var)],
                    cont: Box::new(convert_block(body, &elem_env, ctx, &|_v, body_end_env| {
                        emit_call(&add_unit, vec![CVal::Var(i_var), CVal::Int(1)], idx_ty.clone(), ctx, body_end_env, &|next_i, incr_env| {
                            let mut args = vec![next_i];
                            args.extend(gather_carried(&carried, incr_env));
                            CExpr::App { func: CVal::Label(loop_label.clone()), args }
                        })
                    })),
                };
                let else_cexpr = k(CVal::Unit, cond_env);
                CExpr::If { cond: cond_val, then_branch: Box::new(then_cexpr), else_branch: Box::new(else_cexpr) }
            });

            let mut params = vec![i_var];
            params.extend(carried.iter().map(|(_, v)| *v));
            let mut init_args = vec![CVal::Int(0)];
            init_args.extend(gather_carried(&carried, env));
            let mut all_carried_types = vec![idx_ty.clone()];
            all_carried_types.extend(carried_types);

            CExpr::Fix {
                defs: vec![CFunDef { name: loop_label.clone(), params, body: cond_check, carried_types: Some(all_carried_types) }],
                body: Box::new(CExpr::App { func: CVal::Label(loop_label), args: init_args }),
            }
        }),
        // Collapses a whole chain of nested `Index` nodes into one combined,
        // multi-index `Load` -- see `StmtKind::Assign`'s own `Index` arm for
        // why (the same reasoning applies to reads, for consistency, even
        // though a read alone has no aliasing hazard either way). Only for
        // a real array base -- checked *before* `collect_index_chain` ever
        // walks the chain, on `base`'s own direct type, not the (possibly
        // multi-level) chain-flattened one: chaining is an array-only
        // optimization, never meaningful for the `Index`-algebra fallback
        // just below (an algebra-dispatched result chained through another
        // `[...]` would need its own, separate `Index` impl on `Elem` --
        // real, natural, and not attempted here).
        ExprKind::Index(base, _indices) if matches!(ctx.node_types[&base.id], Ty::Array(..)) => {
            let (array_expr, index_exprs) = collect_index_chain(expr);
            let array_ty = ctx.node_types[&array_expr.id].clone();
            convert_expr(array_expr, env, ctx, &|array_val, env| {
                convert_expr_list(&index_exprs, env, ctx, &|index_vals, env| {
                    let var = ctx.fresh.var();
                    let mut args = vec![array_val.clone()];
                    args.extend(index_vals);
                    CExpr::LetPrim {
                        var,
                        ty: ctx.node_types[&expr.id].clone(),
                        op: PrimOp::Load { array_ty: array_ty.clone() },
                        args,
                        cont: Box::new(k(CVal::Var(var), env)),
                    }
                })
            })
        }
        // `Index<Container, Elem, const K: i32>` algebra dispatch (see
        // `infer.rs`'s own `ExprKind::Index` fallback doc comment) -- an
        // ordinary two-argument algebra call, resolved through the exact
        // same `resolve_call`/`emit_call` machinery any other bare-name
        // algebra call already uses, just without a real `ExprKind::Call`
        // AST node to read a callee name/argument list off of directly
        // (`ExprKind::Index` stays its own dedicated AST shape — needed for
        // the mutability-checking `a[i] = x` requires, see `StmtKind::
        // Assign`'s own `Index` arm). The whole bracket group's own indices
        // become *one* real `[i32;K]` array value -- an ordinary `PrimOp::
        // Array` `LetPrim` built directly from the already-converted index
        // `CVal`s, no different from how any other synthesized intermediate
        // value gets built here -- fed to `index(...)` as its second
        // argument; unpacking it back inside the impl body (`idx[0]`, `idx
        // [1]`, ...) is ordinary array reads, already fully working, no new
        // lowering needed for either half.
        ExprKind::Index(base, indices) => {
            let result_ty = ctx.node_types[&expr.id].clone();
            let index_exprs: Vec<&Expr> = indices.iter().collect();
            let idx_array_ty = Ty::Array(Box::new(Ty::Con("i32".to_string())), Box::new(Ty::Const(ConstValue::Int(indices.len() as u64))));
            convert_expr(base, env, ctx, &|base_val, env| {
                convert_expr_list(&index_exprs, env, ctx, &|index_vals, env| {
                    let idx_array_var = ctx.fresh.var();
                    let callee = resolve_call("index", expr.id, &[base.id], ctx);
                    CExpr::LetPrim {
                        var: idx_array_var,
                        ty: idx_array_ty.clone(),
                        op: PrimOp::Array,
                        args: index_vals,
                        cont: Box::new(emit_call(
                            callee,
                            vec![base_val.clone(), CVal::Var(idx_array_var)],
                            result_ty.clone(),
                            ctx,
                            env,
                            k,
                        )),
                    }
                })
            })
        }
        ExprKind::ArrayLit(elems) => {
            let elem_refs: Vec<&Expr> = elems.iter().collect();
            convert_expr_list(&elem_refs, env, ctx, &|elem_vals, env| {
                let var = ctx.fresh.var();
                CExpr::LetPrim {
                    var,
                    ty: ctx.node_types[&expr.id].clone(),
                    op: PrimOp::Array,
                    args: elem_vals,
                    cont: Box::new(k(CVal::Var(var), env)),
                }
            })
        }
        ExprKind::ArrayRepeat { value, count } => convert_expr(value, env, ctx, &|value_val, env| {
            convert_expr(count, env, ctx, &|count_val, env| {
                let var = ctx.fresh.var();
                CExpr::LetPrim {
                    var,
                    ty: ctx.node_types[&expr.id].clone(),
                    op: PrimOp::ArrayRepeat,
                    args: vec![value_val.clone(), count_val],
                    cont: Box::new(k(CVal::Var(var), env)),
                }
            })
        }),
        // `v.method(args)` — `base` fills the method's own first parameter,
        // an ordinary explicit positional argument, not a magic `self`
        // (`infer.rs`'s own `ExprKind::MethodCall` handling already treats
        // it that way at the type level) — so it converts exactly like any
        // other real call, `[base, ...args]` evaluated left-to-right, just
        // resolved through `resolve_method_call` (a struct's own method
        // namespace, entirely separate from `resolve_call`'s three tiers:
        // `Registry::inherent_method`'s own doc comment guarantees at most
        // one method of a given name per struct, so there's no ambiguity to
        // resolve structurally the way an algebra call needs — either a
        // direct `call_names` override, for a specialization built from a
        // generic inherent impl, or the bare `struct::method` name directly).
        ExprKind::MethodCall(base, name, args) => {
            let struct_ty = ctx.node_types[&base.id].clone();
            let struct_name = match &struct_ty {
                Ty::Con(n) | Ty::App(n, _) => n.clone(),
                other => panic!("CPS: method call on a non-struct type {other:?} -- infer.rs should have rejected this already"),
            };
            let result_ty = ctx.node_types[&expr.id].clone();
            let mut all_args: Vec<&Expr> = vec![base.as_ref()];
            all_args.extend(args.iter());
            convert_expr_list(&all_args, env, ctx, &|arg_vals, env| {
                let callee = resolve_method_call(&struct_name, name, expr, ctx);
                emit_call(callee, arg_vals, result_ty.clone(), ctx, env, k)
            })
        }
        // A standalone block-as-expression (`{ let y = 1; y + 1 }`, reached
        // via `primary`'s own `block` alternative -- or synthesized by
        // `lower.rs`'s own direct-lambda-literal-call desugaring) — was
        // simply never given an arm here before, unlike `if`/`while`/`for`/
        // lambda bodies, which each already call `convert_block` directly
        // off their own dedicated `Block` field. `convert_block` needs
        // nothing new: it already threads its own cloned `env` through the
        // block's own statements and hands the tail's value to `k`, exactly
        // the shape a bare expression position needs.
        ExprKind::Block(b) => convert_block(b, env, ctx, k),
        other => panic!("CPS doesn't support {other:?} yet -- see doc/backlog.md"),
    }
}

/// Walks a chain of nested `Index` nodes, each carrying its own *group* of
/// one or more indices (`a[i,j]`'s own single-node, two-index group; a
/// literal `a[i][j]` — two separate bracket pairs — chains two one-index
/// groups instead, still reaching the same array element either way, see
/// the module's own "Arrays" doc comment) down to the innermost non-`Index`
/// base, returning it alongside every index expression collected along the
/// way, flattened back into the original *source* order (`a[i,j]` ->
/// `(a, [i, j])`; `a[i][j]` -> `(a, [i, j])` too — the walk collects whole
/// groups outside-in, so the *group* order is reversed before flattening,
/// while each group's own internal (already-source-order) index order stays
/// untouched).
fn collect_index_chain(expr: &Expr) -> (&Expr, Vec<&Expr>) {
    let mut groups: Vec<&Vec<Expr>> = Vec::new();
    let mut current = expr;
    while let ExprKind::Index(base, group) = &current.kind {
        groups.push(group);
        current = base;
    }
    groups.reverse();
    let indices: Vec<&Expr> = groups.into_iter().flat_map(|g| g.iter()).collect();
    (current, indices)
}

/// Every enclosing-scope name a branch's/loop's own join/recursive
/// continuation must carry as an extra argument alongside its "value" one —
/// see the module's own "Mutation across control flow" doc comment.
fn tail_call_join(branch_env: &CEnv, value: CVal, carried: &[(String, CVar)], join_label: &str) -> CExpr {
    let mut args = vec![value];
    args.extend(gather_carried(carried, branch_env));
    CExpr::App { func: CVal::Label(join_label.to_string()), args }
}

fn gather_carried(carried: &[(String, CVar)], env: &CEnv) -> Vec<CVal> {
    carried
        .iter()
        .map(|(name, _)| env.get(name).cloned().unwrap_or_else(|| panic!("CPS: `{name}` unexpectedly unbound at a branch/loop join")))
        .collect()
}

/// Every name from an *enclosing* scope that `block` might reassign via a
/// bare-path `StmtKind::Assign`, transitively through any nested `if`/
/// `while`/`for`, mapped to its own concrete type (`ctx.node_types`, read
/// off the assignment *target*'s own node — always present, type-checking
/// annotates it to validate the assigned value's type against it) — a
/// purely syntactic walk (no dataflow fixpoint needed, the structured AST
/// already says everything relevant) — see the module's own "Mutation
/// across control flow" doc comment, and `CFunDef::carried_types`'s own doc
/// comment for why a *type* is needed here at all, not just the name. A name
/// `let`/`let mut`-bound *inside* `block` itself shadows any same-named
/// outer variable for the rest of that scope and is correctly excluded. An
/// indexed/field assignment target doesn't name a scalar to thread at all
/// (Stage 5, not handled here).
fn mutated_free_vars(block: &Block, shadowed: &HashSet<String>, ctx: &Ctx) -> HashMap<String, Ty> {
    let mut local_shadowed = shadowed.clone();
    let mut escaping = HashMap::new();
    for stmt in &block.stmts {
        match &stmt.kind {
            StmtKind::Let { name, value, .. } => {
                escaping.extend(mutated_free_vars_expr(value, &local_shadowed, ctx));
                local_shadowed.insert(name.clone());
            }
            StmtKind::Assign { target, value } => {
                escaping.extend(mutated_free_vars_expr(value, &local_shadowed, ctx));
                if let ExprKind::Path(p) = &target.kind {
                    let name = p.segments.join("::");
                    if !local_shadowed.contains(&name) {
                        escaping.insert(name, ctx.node_types[&target.id].clone());
                    }
                }
            }
            StmtKind::Expr(e) => escaping.extend(mutated_free_vars_expr(e, &local_shadowed, ctx)),
        }
    }
    if let Some(tail) = &block.tail {
        escaping.extend(mutated_free_vars_expr(tail, &local_shadowed, ctx));
    }
    escaping
}

fn mutated_free_vars_expr(expr: &Expr, shadowed: &HashSet<String>, ctx: &Ctx) -> HashMap<String, Ty> {
    match &expr.kind {
        ExprKind::NumberLit { .. } | ExprKind::ImaginaryLit { .. } | ExprKind::BoolLit(_) | ExprKind::Path(_) => HashMap::new(),
        ExprKind::Call(_, _, args, ..) => args.iter().flat_map(|a| mutated_free_vars_expr(a, shadowed, ctx)).collect(),
        ExprKind::FieldAccess(base, _) => mutated_free_vars_expr(base, shadowed, ctx),
        ExprKind::MethodCall(base, _, args) => {
            let mut out = mutated_free_vars_expr(base, shadowed, ctx);
            out.extend(args.iter().flat_map(|a| mutated_free_vars_expr(a, shadowed, ctx)));
            out
        }
        ExprKind::Index(base, indices) => {
            let mut out = mutated_free_vars_expr(base, shadowed, ctx);
            out.extend(indices.iter().flat_map(|i| mutated_free_vars_expr(i, shadowed, ctx)));
            out
        }
        ExprKind::ArrayLit(elems) => elems.iter().flat_map(|e| mutated_free_vars_expr(e, shadowed, ctx)).collect(),
        ExprKind::ArrayRepeat { value, count } => {
            let mut out = mutated_free_vars_expr(value, shadowed, ctx);
            out.extend(mutated_free_vars_expr(count, shadowed, ctx));
            out
        }
        ExprKind::StructLit(_, _, fields) => fields.iter().flat_map(|(_, v)| mutated_free_vars_expr(v, shadowed, ctx)).collect(),
        ExprKind::If { cond, then_branch, else_branch } => {
            let mut out = mutated_free_vars_expr(cond, shadowed, ctx);
            out.extend(mutated_free_vars(then_branch, shadowed, ctx));
            if let Some(eb) = else_branch {
                match &**eb {
                    ElseBranch::If(e) => out.extend(mutated_free_vars_expr(e, shadowed, ctx)),
                    ElseBranch::Block(b) => out.extend(mutated_free_vars(b, shadowed, ctx)),
                }
            }
            out
        }
        ExprKind::While { cond, body } => {
            let mut out = mutated_free_vars_expr(cond, shadowed, ctx);
            out.extend(mutated_free_vars(body, shadowed, ctx));
            out
        }
        ExprKind::For { var, start, end, body } => {
            let mut out = mutated_free_vars_expr(start, shadowed, ctx);
            out.extend(mutated_free_vars_expr(end, shadowed, ctx));
            let mut inner = shadowed.clone();
            inner.insert(var.clone());
            out.extend(mutated_free_vars(body, &inner, ctx));
            out
        }
        ExprKind::ForIn { var, iter, body } => {
            let mut out = mutated_free_vars_expr(iter, shadowed, ctx);
            let mut inner = shadowed.clone();
            inner.insert(var.clone());
            out.extend(mutated_free_vars(body, &inner, ctx));
            out
        }
        ExprKind::Block(b) => mutated_free_vars(b, shadowed, ctx),
        // A lambda's own body isn't walked here -- lambdas aren't converted
        // at all yet (closure conversion is a separate, later pass this
        // module doesn't implement, see its own doc comment), so nothing
        // downstream would ever read a mutation found inside one anyway.
        ExprKind::Lambda { .. } => HashMap::new(),
    }
}

/// Every free variable an `ExprKind::Lambda`'s own body references (any
/// `Path` read, not just an assignment target) — excludes the lambda's own
/// declared params and anything it itself `let`/`for`-binds, mapped to its
/// own concrete type (read off the reference node itself, `node_types`).
/// Reliable even *before* this lambda's own generic instantiation is known
/// (see `monomorphize.rs`'s own lambda-worklist doc comment for the
/// matching reasoning on the specialization side): a captured name was
/// never one of *this* lambda's own quantified scheme variables to begin
/// with (`infer.rs::generalize`'s own `env_fv` exclusion) — its type is
/// already whatever the *enclosing* scope's own already-concrete binding
/// has, unaffected by which instantiation of the lambda itself is being
/// built. Plain `&HashMap<NodeId, Ty>`, not `&Ctx` — called both from here
/// (a real `Ctx` on hand) and from `collect_units` (building a lambda's own
/// `ConcreteUnit`, before any `Ctx` exists yet).
///
/// Unlike `mutated_free_vars_expr`'s own `Lambda` arm (which explicitly
/// stops at a nested lambda — "closure conversion isn't implemented yet,"
/// no longer true here) — this walk *recurses into* a nested lambda's own
/// body: an inner lambda referencing an outer-outer variable needs that
/// variable captured at *this*, the outer, lambda's own level too, so its
/// own generated unit can pass it along as one of the inner unit's own
/// leading capture arguments.
///
/// Consumed at two call sites that must agree on the exact same sorted name
/// order without ever sharing a cache (`collect_units`'s own per-
/// specialization unit-building, and `convert_stmts`'s own per-`let`
/// capture-gathering) — safe because this is a pure, deterministic function
/// of `(params, body)`'s own structure: two calls over the identical AST
/// node produce the identical *name* set every time, regardless of which
/// `node_types` map (a specific specialization's own substituted one, or
/// the enclosing unit's own) happens to be passed for the *type* lookup.
fn lambda_free_vars(params: &[Param], body: &Block, node_types: &HashMap<NodeId, Ty>) -> HashMap<String, Ty> {
    let shadowed: HashSet<String> = params.iter().map(|p| p.name.clone()).collect();
    lambda_free_vars_block(body, &shadowed, node_types)
}

fn lambda_free_vars_block(block: &Block, shadowed: &HashSet<String>, node_types: &HashMap<NodeId, Ty>) -> HashMap<String, Ty> {
    let mut local_shadowed = shadowed.clone();
    let mut free = HashMap::new();
    for stmt in &block.stmts {
        match &stmt.kind {
            StmtKind::Let { name, value, .. } => {
                free.extend(lambda_free_vars_expr(value, &local_shadowed, node_types));
                local_shadowed.insert(name.clone());
            }
            StmtKind::Assign { target, value } => {
                free.extend(lambda_free_vars_expr(target, &local_shadowed, node_types));
                free.extend(lambda_free_vars_expr(value, &local_shadowed, node_types));
            }
            StmtKind::Expr(e) => free.extend(lambda_free_vars_expr(e, &local_shadowed, node_types)),
        }
    }
    if let Some(tail) = &block.tail {
        free.extend(lambda_free_vars_expr(tail, &local_shadowed, node_types));
    }
    free
}

fn lambda_free_vars_expr(expr: &Expr, shadowed: &HashSet<String>, node_types: &HashMap<NodeId, Ty>) -> HashMap<String, Ty> {
    match &expr.kind {
        ExprKind::NumberLit { .. } | ExprKind::ImaginaryLit { .. } | ExprKind::BoolLit(_) => HashMap::new(),
        ExprKind::Path(p) => {
            let name = p.segments.join("::");
            if shadowed.contains(&name) {
                return HashMap::new();
            }
            match node_types.get(&expr.id) {
                Some(ty) => HashMap::from([(name, ty.clone())]),
                // A const-generic reference (`N` in `[v; N]`), or some other
                // name this particular `node_types` map doesn't cover -- not
                // a real captured *value* either way (mirrors `convert_
                // expr`'s own `ExprKind::Path` arm's identical const-generic
                // case), nothing to capture.
                None => HashMap::new(),
            }
        }
        ExprKind::Call(_, _, args, ..) => args.iter().flat_map(|a| lambda_free_vars_expr(a, shadowed, node_types)).collect(),
        ExprKind::FieldAccess(base, _) => lambda_free_vars_expr(base, shadowed, node_types),
        ExprKind::MethodCall(base, _, args) => {
            let mut out = lambda_free_vars_expr(base, shadowed, node_types);
            out.extend(args.iter().flat_map(|a| lambda_free_vars_expr(a, shadowed, node_types)));
            out
        }
        ExprKind::Index(base, indices) => {
            let mut out = lambda_free_vars_expr(base, shadowed, node_types);
            out.extend(indices.iter().flat_map(|i| lambda_free_vars_expr(i, shadowed, node_types)));
            out
        }
        ExprKind::ArrayLit(elems) => elems.iter().flat_map(|e| lambda_free_vars_expr(e, shadowed, node_types)).collect(),
        ExprKind::ArrayRepeat { value, count } => {
            let mut out = lambda_free_vars_expr(value, shadowed, node_types);
            out.extend(lambda_free_vars_expr(count, shadowed, node_types));
            out
        }
        ExprKind::StructLit(_, _, fields) => fields.iter().flat_map(|(_, v)| lambda_free_vars_expr(v, shadowed, node_types)).collect(),
        ExprKind::If { cond, then_branch, else_branch } => {
            let mut out = lambda_free_vars_expr(cond, shadowed, node_types);
            out.extend(lambda_free_vars_block(then_branch, shadowed, node_types));
            if let Some(eb) = else_branch {
                match &**eb {
                    ElseBranch::If(e) => out.extend(lambda_free_vars_expr(e, shadowed, node_types)),
                    ElseBranch::Block(b) => out.extend(lambda_free_vars_block(b, shadowed, node_types)),
                }
            }
            out
        }
        ExprKind::While { cond, body } => {
            let mut out = lambda_free_vars_expr(cond, shadowed, node_types);
            out.extend(lambda_free_vars_block(body, shadowed, node_types));
            out
        }
        ExprKind::For { var, start, end, body } => {
            let mut out = lambda_free_vars_expr(start, shadowed, node_types);
            out.extend(lambda_free_vars_expr(end, shadowed, node_types));
            let mut inner = shadowed.clone();
            inner.insert(var.clone());
            out.extend(lambda_free_vars_block(body, &inner, node_types));
            out
        }
        ExprKind::ForIn { var, iter, body } => {
            let mut out = lambda_free_vars_expr(iter, shadowed, node_types);
            let mut inner = shadowed.clone();
            inner.insert(var.clone());
            out.extend(lambda_free_vars_block(body, &inner, node_types));
            out
        }
        ExprKind::Block(b) => lambda_free_vars_block(b, shadowed, node_types),
        ExprKind::Lambda { params, body, .. } => {
            let mut inner = shadowed.clone();
            inner.extend(params.iter().map(|p| p.name.clone()));
            lambda_free_vars_block(body, &inner, node_types)
        }
    }
}

/// Sorts a `lambda_free_vars` result's own names deterministically — the
/// order both `collect_units` (building a lambda unit's own leading capture
/// parameters) and `convert_stmts` (gathering each capture's own `CVal` at
/// the `let`) must agree on, with no shared cache between them (see
/// `lambda_free_vars`'s own doc comment for why recomputing independently
/// is still safe).
fn sorted_capture_names(captures: &HashMap<String, Ty>) -> Vec<String> {
    let mut names: Vec<String> = captures.keys().cloned().collect();
    names.sort();
    names
}

/// Resolves the callee for a call the CPS conversion itself needs to
/// synthesize (a `for` loop's own implicit bound check/increment) rather
/// than one that appears as a real `Call` node in the source — so there's no
/// `Expr` to key `call_names` off of, and this can only ever go through
/// `ctx.call_index`'s own structural `(name, arg types, return type)` lookup
/// (see `resolve_call`'s own doc comment, tier 3) — always a concrete
/// `Ring`/`Ord` intrinsic in practice, but resolved the same general way
/// regardless.
fn resolve_synthetic_binop<'a>(name: &str, ty: &Ty, ret_ty: &Ty, ctx: &Ctx<'a>) -> &'a str {
    let key = (name.to_string(), vec![ty.to_string(), ty.to_string()], ret_ty.to_string());
    match ctx.call_index.get(&key) {
        Some(unit_name) => ctx.units.get_key_value(unit_name.as_str()).map(|(k, _)| k.as_str()).unwrap(),
        // A known, currently-open gap for a `for`-loop whose *bound itself*
        // names a const generic (`for i in 0..N`): `ty` here can come back
        // as `Ty::Const(<N's own concrete value>)` (a *value*) rather than
        // `Ty::Con(<N's declared width>)` (a *type*) -- `start`'s own type
        // variable gets unified with `end`'s (the const generic's own
        // tracked one) at declaration time, and nothing bridges that back to
        // the declared width the way `check_pending_constraints`'s own
        // `const_widths` map does for the constraint-checking path
        // specifically. See `doc/backlog.md`. Panicking here (rather than
        // guessing a width) is deliberate -- this is a real, unresolved gap,
        // not something to silently paper over.
        None => panic!("CPS: could not resolve implicit `{name}({ty}, {ty}) -> {ret_ty}` needed for a for/while loop bound"),
    }
}

/// Shared by an ordinary source-level `Call` and by a loop's own synthesized
/// bound checks: an `extern` unit becomes a straight-line `LetPrim` (Appel's
/// PRIMOP), a real callee becomes a synthesized continuation plus a tail
/// `App` with it appended (Appel's APP) — see the module's own doc comment.
/// A call never itself mutates `env` (only `Assign` does) — `k` is always
/// invoked with the very same `env` the call started with.
fn emit_call(unit_name: &str, arg_vals: Vec<CVal>, result_ty: Ty, ctx: &Ctx, env: &CEnv, k: &dyn Fn(CVal, &CEnv) -> CExpr) -> CExpr {
    let unit = &ctx.units[unit_name];
    match &unit.body {
        // A real C-ABI call, but straight-line, not `Fix`+`App`: it returns
        // synchronously, same as an MLIR op -- no continuation-passing
        // needed just because the callee happens to live outside cleave
        // entirely.
        UnitBody::Extern(symbol) => {
            let var = ctx.fresh.var();
            CExpr::LetPrim {
                var,
                ty: result_ty,
                op: PrimOp::Extern { symbol: symbol.clone(), param_types: unit.param_types.clone() },
                args: arg_vals,
                cont: Box::new(k(CVal::Var(var), env)),
            }
        }
        // `Derivative` shares `Real`'s own calling convention exactly — by
        // the time any *call site* is converted, `fprime` is semantically
        // an ordinary, real, continuation-passing callee; its own body
        // just isn't attached yet at this point in the pipeline (`UnitBody`'s
        // own doc comment) — nothing here needs to look at that body,
        // only at `unit_name` itself (resolved later, by name).
        UnitBody::Real(_) | UnitBody::Derivative(_) => {
            let result_var = ctx.fresh.var();
            let k_label = ctx.fresh.label("k");
            let mut call_args = arg_vals;
            call_args.push(CVal::Label(k_label.clone()));
            CExpr::Fix {
                defs: vec![CFunDef { name: k_label, params: vec![result_var], body: k(CVal::Var(result_var), env), carried_types: None }],
                body: Box::new(CExpr::App { func: CVal::Label(unit_name.to_string()), args: call_args }),
            }
        }
    }
}

/// Converts a list of expressions left-to-right, sequentially — evaluation
/// order is observable and must stay sequential, never flattened/reordered.
fn convert_expr_list(exprs: &[&Expr], env: &CEnv, ctx: &Ctx, k: &dyn Fn(Vec<CVal>, &CEnv) -> CExpr) -> CExpr {
    fn go(exprs: &[&Expr], env: &CEnv, ctx: &Ctx, acc: Vec<CVal>, k: &dyn Fn(Vec<CVal>, &CEnv) -> CExpr) -> CExpr {
        let Some((first, rest)) = exprs.split_first() else {
            return k(acc, env);
        };
        convert_expr(first, env, ctx, &|v, env| {
            let mut acc2 = acc.clone();
            acc2.push(v);
            go(rest, env, ctx, acc2, k)
        })
    }
    go(exprs, env, ctx, Vec::new(), k)
}

/// See the module's own doc comment ("Resolving a call site's own target
/// unit") for why exactly these three tiers, in this order, are each
/// necessary and together unambiguous.
fn resolve_call<'a>(name: &str, call_id: NodeId, arg_ids: &[NodeId], ctx: &Ctx<'a>) -> &'a str {
    if let Some(mangled) = ctx.call_names.get(&call_id) {
        return ctx.units.get_key_value(mangled.as_str()).map(|(k, _)| k.as_str()).unwrap_or_else(|| {
            panic!("CPS: call_names resolved `{name}` to `{mangled}`, but no such unit exists")
        });
    }
    if let Some((k, _)) = ctx.units.get_key_value(name) {
        return k.as_str();
    }
    let arg_tys: Vec<String> = arg_ids.iter().map(|id| ctx.node_types[id].to_string()).collect();
    let ret_ty = ctx.node_types[&call_id].to_string();
    let key = (name.to_string(), arg_tys, ret_ty);
    match ctx.call_index.get(&key) {
        Some(unit_name) => ctx.units.get_key_value(unit_name.as_str()).map(|(k, _)| k.as_str()).unwrap(),
        None => panic!("CPS: could not resolve call to `{name}` ({key:?})"),
    }
}

/// The `MethodCall` equivalent of `resolve_call` — a separate, simpler
/// resolution namespace: `Registry::inherent_method`'s own doc comment
/// guarantees at most one method of a given name exists per struct, so
/// (unlike an algebra call) there's no structural/signature-based candidate
/// search needed here, just a direct lookup once the struct name is known
/// (already resolved by the caller, off `base`'s own concrete type).
fn resolve_method_call<'a>(struct_name: &str, method: &str, call: &Expr, ctx: &Ctx<'a>) -> &'a str {
    if let Some(mangled) = ctx.call_names.get(&call.id) {
        return ctx.units.get_key_value(mangled.as_str()).map(|(k, _)| k.as_str()).unwrap_or_else(|| {
            panic!("CPS: method call_names resolved `{struct_name}::{method}` to `{mangled}`, but no such unit exists")
        });
    }
    let bare = format!("{struct_name}::{method}");
    ctx.units
        .get_key_value(bare.as_str())
        .map(|(k, _)| k.as_str())
        .unwrap_or_else(|| panic!("CPS: could not resolve method call `{bare}`"))
}

/// A `let`-bound literal is generalized at its own definition site (`let x =
/// 1.0;` gives `x` a polymorphic scheme — see `dump.rs`'s own
/// `dumps_let_and_tail_statements_with_their_own_types` test) — its
/// definition-site `NodeId` in `node_types` can therefore still carry an
/// unresolved `Ty::Var`, even though the *function* is otherwise fully
/// concrete and every *use* of `x` resolves fine (its own, separate `NodeId`
/// gets a fresh, properly-instantiated type). Found by direct testing: `fn f()
/// -> f64 { let x = 1.0; x }` panicked trying to parse `"1.0"` as an int.
/// Falls back to the literal's own textual shape (a `.`/exponent means
/// `Float`, matching exactly how `Infer`'s own number-literal defaulting
/// decides the same question) whenever `node_types` hasn't actually pinned a
/// concrete numeric type here — the concrete-type path stays primary since it
/// alone accounts for an explicit `:f64`-style suffix on integer-shaped text.
/// Materializes a `NumberLit`/`ImaginaryLit` that resolved to `Complex<T>`
/// as a real `PrimOp::Struct` construction — the exact same mechanism
/// `ExprKind::StructLit` already uses, since a `Complex<T>` value has no
/// representation of its own beyond the ordinary heap-allocated struct
/// `stdlib/complex/complex.cleave` declares. `real_text`/`imag_text` are
/// each parsed at `T`'s own concrete width (`elem_ty`) — a plain literal
/// widened to `Complex` passes its own text as `real_text` with `imag_text
/// = "0"`; a bare `4i` does the reverse.
fn complex_literal(ctx: &Ctx, expr: &Expr, real_text: &str, imag_text: &str, env: &CEnv, k: &dyn Fn(CVal, &CEnv) -> CExpr) -> CExpr {
    let ty = ctx.node_types[&expr.id].clone();
    let Ty::App(name, elem_tys) = &ty else {
        panic!("CPS: a Complex-widened literal's own resolved type must be Complex<T>, got {ty}")
    };
    debug_assert_eq!(name, "Complex");
    let elem_ty = &elem_tys[0];
    let real = parse_number(real_text, elem_ty);
    let imag = parse_number(imag_text, elem_ty);
    let var = ctx.fresh.var();
    CExpr::LetPrim {
        var,
        ty,
        op: PrimOp::Struct("Complex".to_string(), vec!["real".to_string(), "imag".to_string()]),
        args: vec![real, imag],
        cont: Box::new(k(CVal::Var(var), env)),
    }
}

fn parse_number(text: &str, ty: &Ty) -> CVal {
    let is_float = match ty {
        Ty::Var(_) => text.contains('.') || text.contains('e') || text.contains('E'),
        _ => matches!(ty.to_string().as_str(), "f32" | "f64"),
    };
    if is_float {
        CVal::Float(text.parse().unwrap_or_else(|e| panic!("bad float literal {text:?}: {e}")))
    } else {
        CVal::Int(text.parse().unwrap_or_else(|e| panic!("bad int literal {text:?}: {e}")))
    }
}

// ---------------------------------------------------------------- dead-code elimination

/// Drops every `CTopLevelFn` unreachable from the program's own real entry
/// point (`"main"`) — `collect_units` unconditionally collects a
/// `ConcreteUnit` for *every* non-generic algebra impl declared anywhere in
/// the merged program, including the whole prelude (`stdlib/num/num.cleave`'s
/// own `Ring`/`Ord` × 6 widths chief among them, see `doc/backlog.md`'s own
/// "Dead-code elimination for unused stdlib specializations" item) —
/// regardless of whether a given program ever actually calls into them.
///
/// Works off already-resolved CPS references (`CVal::Label`), never
/// re-derives call resolution from the AST — a real call's own callee name
/// is already unambiguous by this point (`emit_call`'s own `App{func:
/// Label(name), ..}` shape), matching this module's own "everything past
/// monomorphization is already concrete" discipline throughout. A `Fix`-
/// local continuation's own label (e.g. `"k$0"`) gets swept into the
/// reachable set too along the way — harmless, never collides with a real
/// top-level unit's own name (`"::"`/`"<...>"` vs. `"$"`-delimited naming
/// conventions never overlap), just a few extra no-op entries.
pub fn eliminate_dead_code(program: CpsProgram) -> CpsProgram {
    let by_name: HashMap<&str, &CTopLevelFn> = program.funcs.iter().map(|f| (f.def.name.as_str(), f)).collect();
    let mut reachable: HashSet<String> = HashSet::new();
    let mut worklist: Vec<String> = vec!["main".to_string()];
    while let Some(name) = worklist.pop() {
        if !reachable.insert(name.clone()) {
            continue; // already visited
        }
        if let Some(f) = by_name.get(name.as_str()) {
            collect_called_labels(&f.def.body, &mut worklist);
        }
    }
    CpsProgram { funcs: program.funcs.into_iter().filter(|f| reachable.contains(&f.def.name)).collect() }
}

fn note_label(v: &CVal, out: &mut Vec<String>) {
    if let CVal::Label(name) = v {
        out.push(name.clone());
    }
}

fn collect_called_labels(expr: &CExpr, out: &mut Vec<String>) {
    match expr {
        CExpr::LetPrim { args, cont, .. } => {
            for a in args {
                note_label(a, out);
            }
            collect_called_labels(cont, out);
        }
        CExpr::App { func, args } => {
            note_label(func, out);
            for a in args {
                note_label(a, out);
            }
        }
        CExpr::Fix { defs, body } => {
            for d in defs {
                collect_called_labels(&d.body, out);
            }
            collect_called_labels(body, out);
        }
        CExpr::If { then_branch, else_branch, .. } => {
            collect_called_labels(then_branch, out);
            collect_called_labels(else_branch, out);
        }
    }
}

// ---------------------------------------------------------------- rendering

pub fn dump_cps_program(program: &CpsProgram) -> String {
    let mut out = String::new();
    for (i, f) in program.funcs.iter().enumerate() {
        if i > 0 {
            out.push('\n');
        }
        let params = f.def.params.iter().map(|v| format!("v{v}")).collect::<Vec<_>>().join(" ");
        let _ = writeln!(out, "(fn {} ({params})", f.def.name);
        dump_cexpr(&mut out, &f.def.body, 1);
        let _ = writeln!(out, ")");
    }
    out
}

fn indent(out: &mut String, depth: usize) {
    for _ in 0..depth {
        out.push_str("  ");
    }
}

fn dump_cval(v: &CVal) -> String {
    match v {
        CVal::Var(n) => format!("v{n}"),
        CVal::Int(n) => n.to_string(),
        CVal::Float(n) => n.to_string(),
        CVal::Bool(b) => b.to_string(),
        CVal::Unit => "()".to_string(),
        CVal::Label(name) => name.clone(),
        // Never expected to actually reach a final `CpsProgram` (see the
        // variant's own doc comment) -- rendered distinctly, not panicking,
        // purely so a bug that *did* let one escape shows up legibly in
        // `--dump-cps` output rather than crashing the dumper itself.
        CVal::Closure { captures } => format!("<closure [{}]>", captures.iter().map(dump_cval).collect::<Vec<_>>().join(" ")),
    }
}

fn dump_cexpr(out: &mut String, expr: &CExpr, depth: usize) {
    match expr {
        CExpr::LetPrim { var, ty, op, args, cont } => {
            indent(out, depth);
            let op_str = match op {
                PrimOp::Field { field, .. } => format!("field.{field}"),
                PrimOp::FieldStore { field, .. } => format!("field-store.{field}"),
                PrimOp::Struct(name, fields) => format!("struct.{name}[{}]", fields.join(",")),
                PrimOp::Array => "array".to_string(),
                PrimOp::ArrayRepeat => "array-repeat".to_string(),
                PrimOp::Load { .. } => "load".to_string(),
                PrimOp::Store { .. } => "store".to_string(),
                PrimOp::Extern { symbol, .. } => format!("extern.{symbol}"),
                PrimOp::RawMlirOp { op, attrs } => {
                    let attrs_str: String = attrs.iter().map(|(name, text)| format!(" {name}={text:?}")).collect();
                    format!("mlir.{op}{attrs_str}")
                }
            };
            let args_str = args.iter().map(dump_cval).collect::<Vec<_>>().join(" ");
            let _ = writeln!(out, "(let-prim v{var}: {ty} = ({op_str} {args_str})");
            dump_cexpr(out, cont, depth);
            indent(out, depth);
            out.push_str(")\n");
        }
        CExpr::App { func, args } => {
            indent(out, depth);
            let args_str = args.iter().map(dump_cval).collect::<Vec<_>>().join(" ");
            if args.is_empty() {
                let _ = writeln!(out, "({})", dump_cval(func));
            } else {
                let _ = writeln!(out, "({} {args_str})", dump_cval(func));
            }
        }
        CExpr::Fix { defs, body } => {
            indent(out, depth);
            let _ = writeln!(out, "(fix");
            for d in defs {
                indent(out, depth + 1);
                let params = d.params.iter().map(|v| format!("v{v}")).collect::<Vec<_>>().join(" ");
                let _ = writeln!(out, "({} ({params})", d.name);
                dump_cexpr(out, &d.body, depth + 2);
                indent(out, depth + 1);
                out.push_str(")\n");
            }
            dump_cexpr(out, body, depth + 1);
            indent(out, depth);
            out.push_str(")\n");
        }
        CExpr::If { cond, then_branch, else_branch } => {
            indent(out, depth);
            let _ = writeln!(out, "(if {}", dump_cval(cond));
            dump_cexpr(out, then_branch, depth + 1);
            dump_cexpr(out, else_branch, depth + 1);
            indent(out, depth);
            out.push_str(")\n");
        }
    }
}
