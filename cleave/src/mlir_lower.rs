//! Lowers a CPS-form program (`cps::CpsProgram`) into MLIR, via `melior`.
//!
//! Handles, so far: a top-level function's `return`, straight-line `LetPrim`
//! chains (`extern` calls, reserved `mlir::dialect::op(...)` calls — see
//! `lower_raw_mlir_op`'s own doc comment — array construction/read/write,
//! see `array_memref_type`'s own doc comment, and struct construction/field
//! access, see `struct_llvm_type`'s own doc comment), a two-armed `if`
//! producing a value (`scf.if`), a real call to another top-level cleave
//! `fn` (`func.call`), and a `while`/`for` loop carrying state (`scf.while`,
//! see `lower_loop`'s own doc comment) — together enough for a genuinely
//! recursive function (`fib`) *and* an iterative one (a loop-accumulated
//! sum) to compile and JIT-execute correctly end to end. Grows the same
//! incremental way `cps.rs` itself did: implement exactly what a real
//! example needs, panic clearly on anything else, expand next. Still
//! missing: an `if`/loop whose branches also carry reassigned outer
//! variables *beyond* the loop's own natural carried state (see `lower_if`'s
//! own doc comment).
//!
//! **Type lowering is data-driven, not hardcoded**, matching `doc/hld.md`'s
//! own "one generic 'emit this named MLIR op' primitive" thesis one level up
//! from operations: `ty_to_mlir` looks a cleave type name up in a map built
//! from every `#[mlir_type("...")]`-tagged algebra `impl`
//! (`cps::collect_mlir_types`), parsing the declared MLIR type text via
//! `melior::ir::Type::parse` — no per-type-name Rust match left, beyond
//! `bool`, which stays a genuine special case (matching `infer.rs`'s own
//! hardcoded `Ty::Con("bool")` for `if`/`while` conditions — the *only*
//! other structurally-special type name left anywhere in this compiler).
//!
//! `CFunDef`'s own trailing "return continuation" parameter (`CTopLevelFn::
//! k_ret`) is what makes this tractable without reconstructing real control
//! flow from scratch: a tail call to *exactly* a function's own `k_ret` is
//! that function's own `return`, recognized structurally (`App { func:
//! Var(v), .. } if v == k_ret`). `Fix`/`If` extend the same idea one level
//! further — see `lower_cexpr`'s own `CExpr::Fix` arm.

use crate::ast::Type as AstType;
use crate::ast::{Expr, ExprKind, GenericArg, TypeKind, tuple_struct_name};
use crate::cps::{CExpr, CFunDef, CTopLevelFn, CVal, CVar, CpsProgram, PrimOp, StructSchema};
use crate::infer::{ConstValue, Ty};
use melior::{
    Context,
    dialect::{
        arith, func,
        llvm::{self, LoadStoreOptions},
        memref, scf,
    },
    ir::{
        Attribute, Block, Identifier, Location, Module, Operation, Region, RegionLike, Type,
        TypeLike, Value, ValueLike,
        attribute::{
            DenseI32ArrayAttribute, DenseI64ArrayAttribute, FlatSymbolRefAttribute, FloatAttribute,
            IntegerAttribute, StringAttribute, TypeAttribute,
        },
        block::BlockLike,
        operation::OperationBuilder,
        r#type::{
            DimSize, FunctionType, IntegerType, MemRefType, RankedTensorType, ShapedTypeLike,
        },
    },
};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};

/// What a growing lowering pass needs threaded everywhere. `declared_externs`
/// is a `RefCell` (rather than a `&mut` threaded alongside everything else)
/// because declaring a not-yet-seen `extern` symbol is a cross-cutting side
/// effect on the *module* (appending a new top-level `func.func` declaration
/// to it) that has to happen from deep inside expression lowering — the same
/// shape `ensure_extern_declared` needs, and much simpler than threading a
/// mutable borrow through every recursive call just for this one case.
/// `signatures` is built once, up front, from every top-level `fn` in the
/// program (`program.funcs` — self-recursion works fine since a function's
/// own entry lands here before its own body is ever lowered) — needed
/// because a real call's own `Fix`-synthesized join continuation (see
/// `lower_real_call`) carries no cleave-level type of its own, unlike a
/// `CTopLevelFn`; the callee's own signature is the only place left to get
/// its argument/result types from.
struct LowerCtx<'c, 'm> {
    context: &'c Context,
    module: &'m Module<'c>,
    declared_externs: RefCell<HashSet<String>>,
    signatures: HashMap<String, (Vec<Ty>, Ty)>,
    /// Cleave type name -> MLIR type text, from every `#[mlir_type(...)]`-
    /// tagged algebra `impl` (`cps::collect_mlir_types`) — see this file's
    /// own module doc comment.
    mlir_types: HashMap<String, String>,
    /// Every `struct`'s own declared shape (`cps::collect_struct_schemas`) —
    /// see `struct_llvm_type`'s own doc comment.
    struct_schemas: HashMap<String, StructSchema>,
    /// Every top-level function name `region_analysis::find_region_local_
    /// functions` proved safe to lower with `cleave_alloc_local` at each of
    /// its own construction sites — see that module's own doc comment for
    /// the real analysis, and `alloc_llvm_value`'s own doc comment for how
    /// this set actually gets consulted.
    region_local_fns: HashSet<String>,
    /// Whether the top-level function *currently* being lowered
    /// (`lower_top_level_fn`, which sets this once per function, before
    /// lowering that function's own body) is in `region_local_fns` — a
    /// `Cell`, not threaded as an ordinary parameter, for the same reason
    /// `declared_externs` above is a `RefCell`: a cross-cutting fact about
    /// *which function's body* is presently being lowered, needed deep
    /// inside `alloc_llvm_value` (many call frames down from `lower_top_
    /// level_fn` itself), not something worth threading as an explicit
    /// parameter through every intervening `lower_cexpr`/`lower_prim_op`
    /// call. Never actually re-entrant (this project lowers one top-level
    /// function's own body at a time, start to finish, before moving to the
    /// next), so a plain `Cell<bool>` — not a stack — is enough.
    currently_region_local: std::cell::Cell<bool>,
}

/// Builds one MLIR `Module` containing every top-level function in
/// `program`. `mlir_types`/`struct_schemas` are `cps::collect_mlir_types`/
/// `cps::collect_struct_schemas`'s own output, run against the *original*
/// `ast::Program` — kept as separate parameters rather than folded into
/// `CpsProgram` itself, since both are whole-program, type-level facts with
/// no per-function home, unlike everything else `CpsProgram` carries.
pub fn lower_program<'c>(
    context: &'c Context,
    program: &CpsProgram,
    mlir_types: &HashMap<String, String>,
    struct_schemas: HashMap<String, StructSchema>,
) -> Module<'c> {
    let location = Location::unknown(context);
    let module = Module::new(location);
    let signatures = program
        .funcs
        .iter()
        .map(|f| {
            (
                f.def.name.clone(),
                (f.param_types.clone(), f.result.clone()),
            )
        })
        .collect();
    let region_local_fns = crate::region_analysis::find_region_local_functions(program);
    {
        // Scoped so `ctx`'s own borrow of `module` ends before `module` is
        // moved out below.
        let ctx = LowerCtx {
            context,
            module: &module,
            declared_externs: RefCell::new(HashSet::new()),
            signatures,
            mlir_types: mlir_types.clone(),
            struct_schemas,
            region_local_fns,
            currently_region_local: std::cell::Cell::new(false),
        };
        for f in &program.funcs {
            let op = lower_top_level_fn(&ctx, f);
            ctx.module.body().append_operation(op);
        }
    }
    module
}

/// Maps a cleave `Ty` to its MLIR equivalent. `bool` is a genuine special
/// case (matching `infer.rs`'s own hardcoded `Ty::Con("bool")` for `if`/
/// `while` conditions — the only other structurally-special type name left
/// anywhere in this compiler); every other name is looked up in
/// `ctx.mlir_types` and parsed via `Type::parse` — panics clearly if a type
/// actually needed was never declared `#[mlir_type(...)]` anywhere, rather
/// than guessing.
fn ty_to_mlir<'c>(ctx: &LowerCtx<'c, '_>, ty: &Ty) -> Type<'c> {
    match ty {
        // A struct tagged `#[mlir_type(tensor)]`/`#[mlir_type(vector)]` —
        // checked *before* the ordinary primitive-width lookup just below
        // (whose own `Type::parse` would choke on the bare keyword
        // "tensor"/"vector", not a real, complete type text on its own) and
        // before the generic struct-is-an-opaque-pointer fallback. See
        // `tagged_struct_native_type`'s own doc comment for the full design.
        Ty::Con(name) | Ty::App(name, _) if native_shape_keyword(ctx, name).is_some() => {
            tagged_struct_native_type(ctx, ty)
        }
        Ty::Con(name) if name == "bool" || ctx.mlir_types.contains_key(name) => width_ty(ctx, name),
        // A `Ty::Con`/`Ty::App` that *isn't* a declared primitive (and isn't
        // shape-tagged, handled above) is an ordinary struct — non-generic
        // (`Ty::Con("Vec2")`) or generic-and-instantiated (`Ty::App(
        // "Complex", [Con("f32")])`). A struct value is always an opaque
        // `!llvm.ptr` — see `struct_llvm_type`'s own doc comment for why
        // (reference, not value, semantics).
        Ty::Con(_) | Ty::App(..) => llvm::r#type::pointer(ctx.context, 0),
        // Two representations, picked by the array's own *leaf* element
        // type — see `array_leaf_is_struct`'s own doc comment for why: a
        // struct-typed leaf can't be a `memref` element (`MemRefType::new`
        // rejects it with a hard native assertion, found by direct
        // testing, not a clean `Result`), so it gets the same "opaque
        // `!llvm.ptr`" treatment a struct value already gets everywhere
        // else, pointing at a real heap allocation instead of a `memref`
        // descriptor (`lower_array_construct`/`lower_array_repeat` build
        // it; `lower_array_load`/`lower_array_store` already know how to
        // read one, since a struct's own array-typed *field* already hands
        // back exactly this shape).
        Ty::Array(..) => {
            if array_leaf_is_struct(ctx, ty) {
                llvm::r#type::pointer(ctx.context, 0)
            } else {
                array_memref_type(ctx, ty).into()
            }
        }
        // A bare, resolved const-generic value reaching an ordinary type
        // position — e.g. a turbofish-pinned `const N: i32`'s own call-site
        // result type, `Ty::Const(Int(3))` (see `infer.rs`'s own new `unify`
        // arms reconciling this against an ordinary `Ty::Con`). Widened to
        // its own natural primitive type: `cps.rs`'s own `ExprKind::Path`
        // handling already lowers the *value* side of this identical shape
        // (`Ty::Const(ConstValue::Int(n)) => CVal::Int(n)`) — this is the
        // missing type-side counterpart. No declared width survives in a
        // bare `ConstValue::Int` (see `ConstValue`'s own doc comment) — `i32`
        // is the same default this codebase already uses for an otherwise-
        // unconstrained integer literal.
        Ty::Const(ConstValue::Int(_)) => width_ty(ctx, "i32"),
        Ty::Const(ConstValue::Bool(_)) => width_ty(ctx, "bool"),
        // Should never actually be reached: a `Ty::ConstExpr` (`doc/
        // backlog.md`'s own "Deferred/symbolic constant folding" item) only
        // stays unresolved while its own operands do — by the time
        // monomorphization has produced a real, concrete specialization for
        // codegen to lower at all, `substitute` has already folded it into a
        // plain `Ty::Const` (see that function's own doc comment). A
        // dedicated panic message here, rather than falling into the
        // generic one below, points straight at the real cause if this
        // invariant is ever violated.
        //
        // One real, known way to reach this despite that invariant: a
        // divide-by-zero *inside a still-generic declaration* (e.g. `fn f
        // <const N, const M>() { let x: [T; N/M]; }`, only ever called with
        // `M = 0` at one specific instantiation) — `infer.rs`'s own
        // `pending_div_by_zero_checks` only catches the *immediate*, already-
        // concrete case (a literal or turbofish-pinned divisor), not this
        // deferred one (`Ty` itself carries no source span for `fold_const_
        // expr`/`substitute`, deep in generic-instantiation machinery, to
        // attach a real diagnostic to — `doc/backlog.md`'s own note on this).
        // Named specifically when recognized, rather than the generic
        // message below, so this doesn't read as "the invariant broke" when
        // it's actually a real, if unlocated, division by zero.
        Ty::ConstExpr(op, _, b)
            if op == "div" && matches!(b.as_ref(), Ty::Const(ConstValue::Int(0))) =>
        {
            panic!(
                "MLIR lowering: division by zero in a const-generic expression (`{ty}`), only detected this late because it happens inside a still-generic declaration at one specific instantiation — no source location available here; `doc/backlog.md`'s own note on this"
            )
        }
        Ty::ConstExpr(..) => {
            panic!(
                "MLIR lowering: unresolved deferred const expression `{ty}` reached codegen — should have been folded by `substitute` during monomorphization"
            )
        }
        _ => panic!(
            "MLIR lowering doesn't support type `{ty}` yet (only primitive Ty::Con widths, arrays, and structs so far)"
        ),
    }
}

/// Whether `ty` (an array type) has a struct-typed *leaf* element — the
/// same primitive-vs-struct distinction `ty_to_mlir`'s own `Ty::Con`/
/// `Ty::App` arms already draw, applied to an array's own innermost element
/// instead of the array itself.
fn array_leaf_is_struct(ctx: &LowerCtx, ty: &Ty) -> bool {
    let (_, leaf_ty) = flatten_array_dims(ty);
    matches!(leaf_ty, Ty::Con(name) if name != "bool" && !ctx.mlir_types.contains_key(name))
        || matches!(leaf_ty, Ty::App(..))
}

/// Flattens a nested `Ty::Array(elem, size)` chain — cleave's own multi-dim
/// array type is always nested single-dim arrays, never a separate
/// primitive (`Array(Array(T,C),R)`, see `cps.rs`'s own "Arrays" doc
/// comment) — into *one* flat, multi-dimensional `memref<d0 x d1 x ... x T>`,
/// matching how `cps.rs`'s own `collect_index_chain` already collapses a
/// multi-dim `a[i,j]` access into one combined multi-index `Load`/`Store`
/// against exactly this flat shape: there's never a nested-memref value to
/// address in between. Every dimension is a resolved `Ty::Const(ConstValue::
/// Int(n))` by this stage — cleave has no dynamically-sized arrays.
fn array_memref_type<'c>(ctx: &LowerCtx<'c, '_>, ty: &Ty) -> MemRefType<'c> {
    let (dims, leaf_ty) = flatten_array_dims(ty);
    MemRefType::new(ty_to_mlir(ctx, leaf_ty), &dims, None, None)
}

fn flatten_array_dims(ty: &Ty) -> (Vec<i64>, &Ty) {
    let mut dims = Vec::new();
    let mut cur = ty;
    while let Ty::Array(elem, size) = cur {
        let Ty::Const(ConstValue::Int(n)) = size.as_ref() else {
            panic!("MLIR lowering: array size must be a resolved constant, got `{size}`");
        };
        dims.push(*n as i64);
        cur = elem;
    }
    (dims, cur)
}

fn is_array_ty(ty: &Ty) -> bool {
    matches!(ty, Ty::Array(..))
}

/// Whether `ty` is a `#[mlir_type(tensor)]`/`#[mlir_type(vector)]`-tagged
/// struct (`native_shape_keyword`'s own doc comment) — `None` for anything
/// else, including a plain scalar. Checks the shape first (`struct_name_
/// and_args` panics on a non-struct-shaped `Ty`), unlike every other
/// caller of `native_shape_keyword`, which already knows its own `ty` is
/// struct-shaped going in — this one doesn't (a struct *field*'s own type
/// can be anything).
fn native_shape_field_keyword<'a>(ctx: &'a LowerCtx<'_, '_>, ty: &Ty) -> Option<&'a str> {
    match ty {
        Ty::Con(name) | Ty::App(name, _) => native_shape_keyword(ctx, name),
        _ => None,
    }
}

/// `#[mlir_type(...)]`'s own generalization beyond a bare primitive: the
/// literal keyword `tensor`/`vector` (never itself a complete, parseable
/// MLIR type — that's exactly what makes it safe to distinguish from an
/// ordinary primitive's own real type text, e.g. `#[mlir_type(f32)]`'s
/// `"f32"`) marks a struct whose own sole field — always an array — *is*
/// its real representation, structurally derived, not templated. See
/// `tagged_struct_native_type`'s own doc comment for the full mechanism.
fn native_shape_keyword<'a>(ctx: &'a LowerCtx<'_, '_>, name: &str) -> Option<&'a str> {
    match ctx.mlir_types.get(name).map(String::as_str) {
        s @ (Some("tensor") | Some("vector")) => s,
        _ => None,
    }
}

/// A struct tagged `#[mlir_type(tensor)]`/`#[mlir_type(vector)]`
/// (`stdlib/linalg/tensor.cleave`'s own `Vector<T,N>`/`Matrix<T,R,C>`) —
/// unlike every other struct (`struct_llvm_type`'s own "stable reference,
/// mutated in place" doc comment), this one's real MLIR representation is a
/// native shaped-type SSA *value*, never a heap-allocated opaque pointer.
/// No template string/placeholder substitution needed to know its own
/// shape: the struct's sole field (enforced here — exactly one field,
/// itself array-typed) already carries its own dims/leaf element type,
/// fully understood structurally via the *same* `flatten_array_dims` an
/// ordinary standalone array already uses — `#[mlir_type(...)]` stays
/// exactly as simple a mechanism as it always was (a bare keyword, no text
/// to parse or substitute into), just consulted differently for this one
/// case. This is the real fix for the "closing the loop" gap `Ty::Vector`
/// (a hardcoded new `Ty` variant, touching every exhaustive match in
/// `infer.rs`) used to paper over: `Vector`/`Matrix` are ordinary generic
/// structs, declared entirely in stdlib source, `infer.rs` never needs to
/// know they're anything special at all.
fn tagged_struct_native_type<'c>(ctx: &LowerCtx<'c, '_>, ty: &Ty) -> Type<'c> {
    let (name, type_args) = struct_name_and_args(ty);
    let keyword = native_shape_keyword(ctx, name).unwrap_or_else(|| {
        panic!("MLIR lowering: `{name}` has no recognized #[mlir_type(...)] shape keyword")
    });
    let fields = struct_field_types(&ctx.struct_schemas, name, type_args);
    let [(_, field_ty)] = fields.as_slice() else {
        panic!(
            "MLIR lowering: `#[mlir_type({keyword})]` requires exactly one field, `{name}` has {}",
            fields.len()
        );
    };
    if !matches!(field_ty, Ty::Array(..)) {
        panic!(
            "MLIR lowering: `#[mlir_type({keyword})]`'s sole field must be an array, `{name}`'s own field is `{field_ty}`"
        );
    }
    let (dims, leaf_ty) = flatten_array_dims(field_ty);
    let leaf = ty_to_mlir(ctx, leaf_ty);
    let dims_text = dims.iter().map(|d| format!("{d}x")).collect::<String>();
    Type::parse(ctx.context, &format!("{keyword}<{dims_text}{leaf}>"))
        .unwrap_or_else(|| panic!("MLIR lowering: failed to parse `{keyword}<{dims_text}{leaf}>`"))
}

fn is_unit_ty(ty: &Ty) -> bool {
    matches!(ty, Ty::Con(name) if name == "()")
}

/// A cleave `struct` is a **stable reference, mutated in place** — the same
/// choice `cps.rs`'s own "Arrays" doc comment already makes for arrays, and
/// for the same underlying reason: a struct value flows around by *identity*
/// (assignment, passing to a function, returning it), never copied element
/// by element, and a field write must be visible through every other
/// reference to the same struct. Concretely: every struct-typed cleave value
/// is an opaque `!llvm.ptr` (`ty_to_mlir`'s own struct arm) into one
/// `llvm.alloca`'d slot of this function's own *anonymous* (unnamed, purely
/// structural) `!llvm.struct<(f0_ty, f1_ty, ...)>` — the `llvm` dialect's
/// own aggregate type is the natural fit for a *heterogeneous*-field
/// container (unlike `memref`, which requires one uniform element type
/// across every slot). `llvm.getelementptr` computes each field's own
/// address (`lower_struct_construct`/`lower_field_access`), `llvm.load`/
/// `llvm.store` read/write it — ordinary, generically-typed ops that verify
/// and execute fine even in an otherwise not-yet-`--convert-to-llvm`-lowered
/// module, no different in that respect from `arith`/`scf`/`func`'s own
/// ops. An array-typed field is embedded *inline*, as a (possibly nested)
/// `!llvm.array<N x T>` — not a `memref`, which can't be an `!llvm.struct`
/// field at all (confirmed directly: `Type::parse`+`module.verify()`
/// rejects it, "operand #1 must be primitive LLVM type") — `llvm.
/// getelementptr` walks straight through the struct *and* the nested array
/// in one instruction, so `a.values[i,j]` never needs its own separate
/// allocation, just further indices on the same GEP chain (see
/// `lower_array_load`'s own doc comment). Struct values carry *no name of
/// their own* once lowered — `lower_struct_construct`/`lower_field_access`
/// both need the *cleave-level* struct name (from `PrimOp::Struct`'s own
/// payload, or `PrimOp::Field`'s own `struct_ty`) to resolve field order/
/// types, recovering it from an already-lowered MLIR `Value` alone isn't
/// possible.
fn struct_llvm_type<'c>(ctx: &LowerCtx<'c, '_>, name: &str, type_args: &[Ty]) -> Type<'c> {
    let fields = struct_field_types(&ctx.struct_schemas, name, type_args);
    let field_mlir: Vec<Type> = fields
        .iter()
        .map(|(_, t)| ty_to_llvm_field_type(ctx, t))
        .collect();
    llvm::r#type::r#struct(ctx.context, &field_mlir, false)
}

/// A cleave type as it appears *embedded inside* an `!llvm.struct` field —
/// identical to `ty_to_mlir` for every ordinary scalar/struct case (a
/// nested struct field is, like everything else struct-typed, an opaque
/// `!llvm.ptr`), but an array becomes an *inline* (possibly nested)
/// `!llvm.array<N x T>` rather than a `memref` — see `struct_llvm_type`'s
/// own doc comment for why.
///
/// A `#[mlir_type(tensor)]`-tagged field (`Tensor`) gets a *different* sized
/// LLVM type — `ty_to_mlir` alone would give the *raw* native MLIR type
/// (`tensor<1x2xf32>`, right for a bare function parameter/local, where MLIR
/// natively handles such a value) and that's never a sized LLVM type an
/// `!llvm.struct` field can actually hold — found directly: `'llvm.store' op
/// operand #0 must be LLVM type with size, but got 'tensor<1x2xf32>'`, the
/// moment an ordinary struct (`Dense`/`Network`, `doc/backlog.md`'s own
/// "gradient w.r.t. a struct parameter" item) first tried to wrap a real
/// `Tensor` field. Originally (and for a long stretch of this project,
/// numerically correct throughout) this was the *identical* inline-array
/// treatment an ordinary untagged array field still gets just below — found,
/// much later, by direct re-measurement at real network scale (`doc/backlog.
/// md`'s own "digits-interop" perf item), to be the dominant real compile-
/// time cost: every struct-field crossing a large `Tensor` value makes pays
/// one MLIR op *per element*, in both directions. Now a real memref
/// descriptor instead (`memref_descriptor_llvm_type`'s own doc comment) —
/// `store_native_shape_field`/`load_native_shape_field` are this
/// representation's own O(1) write/read halves.
fn ty_to_llvm_field_type<'c>(ctx: &LowerCtx<'c, '_>, ty: &Ty) -> Type<'c> {
    if let Some(keyword) = native_shape_field_keyword(ctx, ty) {
        let (name, type_args) = struct_name_and_args(ty);
        let fields = struct_field_types(&ctx.struct_schemas, name, type_args);
        let [(_, inner_ty)] = fields.as_slice() else {
            panic!(
                "MLIR lowering: `#[mlir_type({keyword})]` requires exactly one field, `{name}` has {}",
                fields.len()
            );
        };
        // A `Tensor`-tagged field's own storage is a real memref descriptor
        // (`memref_descriptor_llvm_type`'s own doc comment), not an inline
        // flattened array the way an *ordinary* (untagged) array field still
        // is just below — `store_native_shape_field`/`load_native_shape_
        // field` are this representation's own O(1) write/read halves.
        assert_eq!(
            keyword, "tensor",
            "MLIR lowering: O(1) native-shape field storage needs a real memref-backed form, which `#[mlir_type(vector)]` doesn't have"
        );
        let (dims, _leaf_ty) = flatten_array_dims(inner_ty);
        return memref_descriptor_llvm_type(ctx.context, dims.len());
    }
    match ty {
        Ty::Array(..) => {
            let (dims, leaf_ty) = flatten_array_dims(ty);
            let mut t = ty_to_llvm_field_type(ctx, leaf_ty);
            for &d in dims.iter().rev() {
                t = llvm::r#type::array(t, d as u32);
            }
            t
        }
        _ => ty_to_mlir(ctx, ty),
    }
}

/// `name`'s own declared fields, in *declaration* order, each resolved to a
/// concrete `Ty` for *this specific* instantiation (`type_args` — empty for
/// a non-generic struct) — the field-name -> position mapping both struct
/// construction (`llvm.insertvalue`) and field access (`llvm.extractvalue`)
/// need, since neither op's own `position` attribute means anything without
/// a canonical, whole-program-consistent field order.
///
/// Takes `struct_schemas` directly (not `ctx: &LowerCtx`, the only thing it
/// ever read off `ctx`) and is `pub` — `egraph.rs::synthesize_derivatives`
/// reuses this exact function to resolve a struct-typed `derive()` parameter
/// (or one of its own fields) down to its real field shape when building the
/// leaf-enumeration/reassembly machinery for a struct-shaped gradient, the
/// identical need this already served for MLIR lowering.
pub fn struct_field_types(
    struct_schemas: &HashMap<String, StructSchema>,
    name: &str,
    type_args: &[Ty],
) -> Vec<(String, Ty)> {
    let Some(schema) = struct_schemas.get(name) else {
        panic!("MLIR lowering: no struct declaration found for `{name}`");
    };
    // A pack-generic struct (`doc/backlog.md`'s own "Variadic generics"
    // item) needs a genuinely different zip: every non-pack generic 1:1
    // against `type_args`, then *everything remaining* belongs to the
    // pack — `Ty::App`'s own type-args list is already fully flat either
    // way (`Box3<f64,3,4,5>`'s own `[f64,3,4,5]`, no separate "this part is
    // a pack" marker needed there at all, exactly as many entries as the
    // construction site's own turbofish supplied — see `infer.rs::infer_
    // struct_lit_with_pack`'s own doc comment for where that list was
    // built).
    if schema.has_pack {
        let non_pack = &schema.generics[..schema.generics.len() - 1];
        let pack_name = schema
            .generics
            .last()
            .expect("has_pack implies at least one generic");
        let mapping: HashMap<String, Ty> = non_pack
            .iter()
            .cloned()
            .zip(type_args.iter().cloned())
            .collect();
        let pack_tys = &type_args[non_pack.len().min(type_args.len())..];
        schema
            .fields
            .iter()
            .map(|(field_name, field_ast_ty)| {
                (
                    field_name.clone(),
                    resolve_struct_field_ty_with_pack(field_ast_ty, &mapping, pack_name, pack_tys),
                )
            })
            .collect()
    } else {
        let mapping: HashMap<String, Ty> = schema
            .generics
            .iter()
            .cloned()
            .zip(type_args.iter().cloned())
            .collect();
        schema
            .fields
            .iter()
            .map(|(field_name, field_ast_ty)| {
                (
                    field_name.clone(),
                    resolve_struct_field_ty(field_ast_ty, &mapping),
                )
            })
            .collect()
    }
}

/// The pack-aware counterpart to `resolve_struct_field_ty` — identical for
/// every ordinary shape (delegated to directly), plus the same two pack
/// positions `infer.rs::ty_from_ast_mapped_with_pack` already handles at
/// type-checking time: a whole array-dimension list (`[T; Dims...]`,
/// expands to nested `Ty::Array` levels) or a whole field type (`Args...`,
/// becomes the tuple formed from `pack_tys`). Shallow, deliberately, same
/// reasoning as its `infer.rs` counterpart's own doc comment.
fn resolve_struct_field_ty_with_pack(
    ty: &AstType,
    mapping: &HashMap<String, Ty>,
    pack_name: &str,
    pack_tys: &[Ty],
) -> Ty {
    match &ty.kind {
        TypeKind::Array(elem, size) if matches!(&size.kind, ExprKind::PackRef(n) if n == pack_name) =>
        {
            let elem_ty = resolve_struct_field_ty(elem, mapping);
            pack_tys.iter().rev().fold(elem_ty, |acc, dim| {
                Ty::Array(Box::new(acc), Box::new(dim.clone()))
            })
        }
        TypeKind::PackRef(name) if name == pack_name => {
            let tuple_name = tuple_struct_name(pack_tys.len());
            if pack_tys.is_empty() {
                Ty::Con(tuple_name)
            } else {
                Ty::App(tuple_name, pack_tys.to_vec())
            }
        }
        // A pack reference nested *inside* another generic type's own
        // argument list (`m: Tensor<T, Dims...>`, an `AdamState<T,
        // Dims...>` field) — mirrors `infer.rs::ty_from_ast_mapped_with_
        // pack`'s own identical arm (same bug, same fix, found the same
        // way: a struct whose own pack-generic field is itself another
        // pack-generic type, as opposed to `Tensor`'s own bare-array-size
        // `data: [T; Dims...]` or a field that's the pack's own whole
        // type). Every element of `pack_tys` splices into this one
        // argument-list slot (`Tensor<T, Dims...>`'s own concrete
        // dimensions laid out flat, exactly like a direct `Tensor<f32,2,
        // 2>` reference already has), rather than collapsing to a tuple
        // the way the bare-`PackRef`-as-whole-type arm above does —
        // recursing through this same pack-aware function so a deeper
        // nesting resolves correctly too.
        TypeKind::Path(path, args) => {
            let name = path.segments.join("::");
            if args.is_empty() {
                if let Some(mapped) = mapping.get(&name) {
                    return mapped.clone();
                }
                return Ty::Con(name);
            }
            let mut type_args = Vec::with_capacity(args.len());
            for a in args {
                match a {
                    GenericArg::Type(t) if matches!(&t.kind, TypeKind::PackRef(n) if n == pack_name) => {
                        type_args.extend(pack_tys.iter().cloned());
                    }
                    GenericArg::Type(t) => {
                        type_args.push(resolve_struct_field_ty_with_pack(t, mapping, pack_name, pack_tys));
                    }
                    GenericArg::Const(e) => {
                        type_args.push(resolve_struct_field_const(e, mapping));
                    }
                }
            }
            Ty::App(name, type_args)
        }
        _ => resolve_struct_field_ty(ty, mapping),
    }
}

/// A small, standalone counterpart to `infer.rs`'s own `ty_from_ast_mapped`
/// — same core structural recursion (a mapped bare path resolves to the
/// substituted `Ty`, everything else rebuilds the equivalent `Ty` node), but
/// without any of the type-checker's own side effects (`pending_type_name_
/// checks`, fresh-variable allocation for anything not already resolved,
/// ...) — none of which apply this late: every field type a *fully type-
/// checked* program's own struct declarations can produce is already either
/// a mapped generic, a concrete `Path`, or an array over the same, nothing
/// left unresolved to defer.
fn resolve_struct_field_ty(ty: &AstType, mapping: &HashMap<String, Ty>) -> Ty {
    match &ty.kind {
        TypeKind::Path(path, args) => {
            let name = path.segments.join("::");
            if args.is_empty() {
                if let Some(mapped) = mapping.get(&name) {
                    return mapped.clone();
                }
                Ty::Con(name)
            } else {
                let type_args = args
                    .iter()
                    .map(|a| match a {
                        GenericArg::Type(t) => resolve_struct_field_ty(t, mapping),
                        GenericArg::Const(e) => resolve_struct_field_const(e, mapping),
                    })
                    .collect();
                Ty::App(name, type_args)
            }
        }
        TypeKind::Array(elem, size) => Ty::Array(
            Box::new(resolve_struct_field_ty(elem, mapping)),
            Box::new(resolve_struct_field_const(size, mapping)),
        ),
        TypeKind::Fn(..) => {
            panic!("MLIR lowering doesn't support a function-typed struct field yet")
        }
        // `doc/backlog.md`'s own "Variadic generics" item -- grammar/AST
        // exist (Milestone 1), nothing resolves a pack to a concrete list
        // yet, so a struct field can never legitimately still be one by the
        // time a fully type-checked program reaches MLIR lowering.
        TypeKind::PackRef(name) => panic!(
            "MLIR lowering: unresolved pack reference `{name}...` reached a struct field's own type -- variadic generics aren't semantically supported yet"
        ),
    }
}

/// Resolves a struct field's own const-generic-or-literal position (an
/// array field's size, `values: [T; R, C]`'s own `R`/`C`) — every case a
/// fully type-checked program's own struct declarations can actually
/// contain: a bare integer literal, or a bare name mapped to this specific
/// instantiation's own resolved const-generic argument.
fn resolve_struct_field_const(expr: &Expr, mapping: &HashMap<String, Ty>) -> Ty {
    match &expr.kind {
        ExprKind::NumberLit { text, .. } => {
            Ty::Const(ConstValue::Int(text.parse().unwrap_or_else(|_| {
                panic!("MLIR lowering: invalid array-size literal `{text}`")
            })))
        }
        ExprKind::Path(p) if p.segments.len() == 1 => {
            mapping.get(&p.segments[0]).cloned().unwrap_or_else(|| {
                panic!(
                    "MLIR lowering: unresolved const-generic `{}` in a struct field's array size",
                    p.segments[0]
                )
            })
        }
        other => panic!(
            "MLIR lowering doesn't support this struct field array-size expression yet: {other:?}"
        ),
    }
}

/// `ty` is a struct's own concrete type — either `Ty::Con(name)` (no
/// generics) or `Ty::App(name, type_args)` (instantiated) — factored out
/// since `PrimOp::Struct`'s own `LetPrim::ty` carries exactly this, with no
/// separate need to also thread `type_args` through `PrimOp::Struct`'s own
/// payload.
fn struct_name_and_args(ty: &Ty) -> (&str, &[Ty]) {
    match ty {
        Ty::Con(name) => (name.as_str(), &[]),
        Ty::App(name, args) => (name.as_str(), args.as_slice()),
        _ => panic!("MLIR lowering: expected a struct type, got `{ty}`"),
    }
}

/// Like `ty_to_mlir`, but from a bare cleave type name (`"i32"`, `"f64"`,
/// ...) rather than a `Ty` — the form a `PrimOp::RawMlirOp`'s own operand-
/// type inference sometimes needs when working from an already-lowered
/// sibling `Value`'s own MLIR type text isn't available, only its name.
fn width_ty<'c>(ctx: &LowerCtx<'c, '_>, name: &str) -> Type<'c> {
    if name == "bool" {
        return IntegerType::new(ctx.context, 1).into();
    }
    let Some(text) = ctx.mlir_types.get(name) else {
        panic!("MLIR lowering: no `#[mlir_type(...)]` declared for type `{name}`");
    };
    Type::parse(ctx.context, text)
        .unwrap_or_else(|| panic!("MLIR lowering: invalid MLIR type text `{text}` for `{name}`"))
}

/// A `()`-returning top-level `fn` (`main()`, no `->` clause at all — `main`
/// is the only such case any current example actually needs) gets *zero*
/// MLIR results, not "one result of some unit type" — MLIR itself has no
/// value type for "nothing", only "zero results" at the op level (`func.
/// return`/a function's own `FunctionType`). `result_type` below still needs
/// *some* `Type<'c>` to satisfy `lower_cexpr`'s own non-optional parameter,
/// but it's never actually consulted in the unit case: a `()`-typed body's
/// own final `App{k_ret, args}` always has `args == []` (nothing to
/// materialize), so the placeholder is inert. Scoped to *this* function only
/// — a unit-typed function reachable through `lower_real_call`/`PrimOp::
/// Extern` (not just as the program's own entry point) isn't handled yet,
/// and would still panic clearly in `ty_to_mlir` rather than misbehave.
fn lower_top_level_fn<'c>(ctx: &LowerCtx<'c, '_>, f: &CTopLevelFn) -> Operation<'c> {
    let context = ctx.context;
    let location = Location::unknown(context);
    let param_types: Vec<Type> = f.param_types.iter().map(|t| ty_to_mlir(ctx, t)).collect();
    let is_unit = is_unit_ty(&f.result);
    let result_type: Type = if is_unit {
        IntegerType::new(context, 1).into()
    } else {
        ty_to_mlir(ctx, &f.result)
    };
    let results: Vec<Type> = if is_unit { vec![] } else { vec![result_type] };

    let block = Block::new(
        &param_types
            .iter()
            .map(|&t| (t, location))
            .collect::<Vec<_>>(),
    );

    // `f.def.params` is `[ordinary params..., k_ret]` -- `f.param_types`
    // covers only the ordinary ones (see `CTopLevelFn`'s own doc comment),
    // so zip against everything but the last entry.
    let mut env: HashMap<CVar, Value> = HashMap::new();
    for (i, &var) in f.def.params[..f.def.params.len() - 1].iter().enumerate() {
        env.insert(var, block.argument(i).unwrap().into());
    }

    ctx.currently_region_local
        .set(ctx.region_local_fns.contains(&f.def.name));
    lower_cexpr(ctx, &block, env, f.k_ret, result_type, &[], &f.def.body);

    let region = Region::new();
    region.append_block(block);

    // `llvm.emit_c_interface` (and the extra `_mlir_ciface_<name>` wrapper
    // function it generates) is required for `ExecutionEngine::invoke_packed`
    // to find a callable wrapper at all (found by direct testing: without
    // it, JIT invocation fails with "Symbols not found:
    // [ _mlir_ciface_<name> ]" -- `invoke_packed` calls through this
    // specific C-interface wrapper, not the raw function directly) --
    // but the *only* function this project ever calls via `invoke_packed`
    // is `main` itself (confirmed directly: every `--run`/test call site
    // hardcodes `"main"`, never anything else). Attaching it to every
    // top-level fn unconditionally used to double the function count in
    // every compiled module for no functional benefit -- an ordinary
    // internal call already goes through the plain `llvm.call @name`, which
    // never needs this wrapper. An `export fn` (`ast.rs`'s own `FnDecl::
    // is_export` doc comment) does NOT need it either, as long as its
    // signature stays scalar-only (the MVP scope `collect_units` enforces
    // today): a scalar-argument `func.func`'s own LLVM form after
    // `-convert-to-llvm` already has an ordinary, directly C-ABI-callable
    // signature, no memref-descriptor unpacking involved -- unlike
    // `invoke_packed`, a real linked caller (Rust FFI) can call the raw
    // symbol as-is. Revisit once a `Tensor`/struct crosses an `export fn`
    // boundary: *that* case needs the descriptor shape this wrapper exists
    // for, matched by a `#[repr(C)]` type on the Rust side -- not attempted
    // here.
    // `sym_visibility = "private"` on every function that is neither `main`
    // (needs public visibility for `invoke_packed`'s own by-name lookup)
    // nor an `export fn` (needs it for a real external/Rust caller to link
    // against the raw symbol) -- found necessary, not decorative: `--affine-
    // super-vectorize` (`mlir_lower.rs`'s own... no, `pipeline.rs`'s own
    // structured-vectorization stage) hard-errors on a *dead* function
    // still carrying a cross-function-boundary (`strided<[?,?],offset:?>`)
    // memref shape -- `--inline` (the pipeline's own first stage) leaves
    // the original, now-unreferenced declaration of every inlined call
    // behind rather than deleting it, and `--symbol-dce` (run right after)
    // only ever removes a symbol that's *both* unreferenced *and* private —
    // every function used to default to MLIR's own public visibility here,
    // so none of them ever qualified. Every internal algebra-dispatch/
    // helper function is *never* called from outside this module (only
    // `main`/exports are), so `private` costs nothing real.
    let attrs: Vec<(melior::ir::Identifier, melior::ir::Attribute)> = if f.def.name == "main" {
        vec![(
            melior::ir::Identifier::new(context, "llvm.emit_c_interface"),
            melior::ir::Attribute::unit(context),
        )]
    } else if f.is_export {
        vec![]
    } else {
        vec![(
            melior::ir::Identifier::new(context, "sym_visibility"),
            StringAttribute::new(context, "private").into(),
        )]
    };
    // An exported unit's real LLVM symbol is its `export_symbol` override
    // when given, else its own cleave name unchanged. NOTE (known, scoped
    // gap, not silently wrong): overriding the symbol here does *not*
    // rewrite any internal cleave-side call to this same function -- those
    // still resolve to `f.def.name` (`emit_call`'s own `CVal::Label`
    // resolution, upstream of this file). Fine for this MVP's actual target
    // (a leaf kernel exported for an external host, no cleave-side caller of
    // its own) but a real bug if some *other* cleave unit ever called an
    // `export(other_symbol)`-renamed function internally -- not attempted
    // to fix here, would need the internal-call-name resolution itself
    // (`cps.rs`) to keep tracking the unrenamed name separately.
    let symbol_name: &str = if f.is_export {
        f.export_symbol.as_deref().unwrap_or(&f.def.name)
    } else {
        &f.def.name
    };
    func::func(
        context,
        StringAttribute::new(context, symbol_name),
        TypeAttribute::new(FunctionType::new(context, &param_types, &results).into()),
        region,
        &attrs,
        location,
    )
}

/// `env` is owned, not borrowed, and moved through the recursion — a
/// `LetPrim` chain extends it one binding at a time (`env.insert` below),
/// and threading it by value avoids either cloning the whole map per step or
/// fighting the borrow checker over a `&mut` shared with everything else
/// `ctx` already carries. Branching (`lower_if`) does need to hand each arm
/// its own copy, since the two are alternatives, not a sequence — `Value`
/// itself is `Copy`, so cloning the map is cheap.
///
/// `yield_targets`: every `Fix`-synthesized continuation (`if`-join, see
/// `lower_if`, or a loop's own self-recursion, see `lower_loop`) whose own
/// body is *currently* being lowered, outermost first — a stack, not a
/// single slot: a tail `App` to *any* of these names means `scf.yield`
/// against that specific entry's own `types` (one per yielded position,
/// needed since, unlike the enclosing function's own single `result_type`,
/// a join or loop's own carried values aren't necessarily the same type as
/// the function's overall return value), never a real call or a function
/// return. A stack, not the single innermost entry, because of `break`
/// (`doc/backlog-done.md`'s own "break value" item, found directly while
/// implementing it): a `break` nested inside an `if` (itself nested inside
/// the loop it targets) tail-calls the *loop's* own label directly, past
/// the `if`-join sitting between them — recognizing that needs every
/// still-open enclosing target visible, not just the immediately-enclosing
/// one. `lower_if`/`lower_loop` each *push* their own new entry onto this
/// stack (a fresh `Vec` built at that one call site, not a mutation of the
/// caller's own) while lowering their *own* branches/body, then use the
/// original, unpushed slice for whatever runs after the join/loop — ordinary
/// flow resumes there, with that join/loop's own name no longer a valid
/// target. Empty at a function's own top-level body.
///
/// Each entry's own third field (`YieldTarget`'s own doc comment) is `Some
/// (region_handle)` for a *loop's* own self-recursive target (`lower_loop`
/// pushes it, having just called `cleave_region_enter` itself) and `None`
/// for an ordinary `if`-join (`lower_if` never closes a region — only a
/// loop's own tail-recursive "continue" genuinely marks the end of one
/// iteration).
type YieldTarget<'c, 'a> = (&'a str, &'a [Type<'c>], Option<Value<'c, 'c>>);

fn lower_cexpr<'c>(
    ctx: &LowerCtx<'c, '_>,
    block: &Block<'c>,
    mut env: HashMap<CVar, Value<'c, 'c>>,
    k_ret: CVar,
    result_type: Type<'c>,
    yield_targets: &[YieldTarget<'c, '_>],
    expr: &CExpr,
) {
    match expr {
        // `args == [CVal::Unit]` is a `()`-returning function's own
        // "nothing to return" — matches `lower_top_level_fn`'s own zero-
        // result `FunctionType` for the unit case (see its doc comment):
        // `CVal::Unit` is never lowered into a real MLIR value at all, it's
        // filtered out here instead of reaching `lower_cval` (which doesn't
        // support it).
        CExpr::App {
            func: CVal::Var(v),
            args,
        } if *v == k_ret => {
            let location = Location::unknown(ctx.context);
            let values: Vec<Value> = args
                .iter()
                .filter(|a| !matches!(a, CVal::Unit))
                .map(|a| lower_cval(ctx.context, block, &env, a, result_type))
                .collect();
            block.append_operation(func::r#return(&values, location));
        }
        CExpr::App {
            func: CVal::Label(name),
            args,
        } if yield_targets.iter().any(|(n, _, _)| *n == name.as_str()) => {
            let location = Location::unknown(ctx.context);
            let (_, types, region_handle) = yield_targets
                .iter()
                .find(|(n, _, _)| *n == name.as_str())
                .unwrap();
            // `CVal::Unit` filtered out here, same as the `return` arm above
            // and for the same reason -- an `if`-join's own value position
            // is unit for a bodyless-`else`/statement-position `if`
            // (`if cond { mutate; };`), and `types` (built by `lower_if`)
            // already excludes that position too, so the two stay aligned.
            let values: Vec<Value> = args
                .iter()
                .filter(|a| !matches!(a, CVal::Unit))
                .zip(*types)
                .map(|(a, &t)| lower_cval(ctx.context, block, &env, a, t))
                .collect();
            // A loop's own "continue" target carries a real region handle
            // (`YieldTarget`'s own doc comment) -- this iteration's own
            // region closes exactly here, right before yielding the next
            // iteration's carried state, the mirror image of `lower_loop`'s
            // own `cleave_region_enter` at the body's start. An `if`-join's
            // own target never does (`region_handle` is `None`) -- an `if`
            // is not an iteration boundary.
            if let Some(handle) = region_handle {
                ensure_extern_declared(
                    ctx,
                    "cleave_region_exit",
                    &[Ty::Con("i64".to_string())],
                    &[],
                );
                block.append_operation(func::call(
                    ctx.context,
                    FlatSymbolRefAttribute::new(ctx.context, "cleave_region_exit"),
                    &[*handle],
                    &[],
                    location,
                ));
            }
            block.append_operation(scf::r#yield(&values, location));
        }
        CExpr::LetPrim {
            var,
            ty,
            op,
            args,
            cont,
        } => {
            if let Some(value) = lower_prim_op(ctx, block, &env, op, args, ty) {
                env.insert(*var, value);
            }
            lower_cexpr(ctx, block, env, k_ret, result_type, yield_targets, cont);
        }
        CExpr::Fix { defs, body } => {
            let [def] = &defs[..] else {
                panic!(
                    "MLIR lowering doesn't support a multi-def `Fix` yet -- see mlir_lower.rs's own module doc comment"
                );
            };
            match &**body {
                CExpr::If {
                    cond,
                    then_branch,
                    else_branch,
                } => {
                    lower_if(
                        ctx,
                        block,
                        env,
                        k_ret,
                        result_type,
                        yield_targets,
                        def,
                        cond,
                        then_branch,
                        else_branch,
                    );
                }
                // A loop's own *entry*: `def` (the loop's self-recursive
                // continuation) called directly, with no trailing
                // continuation label -- distinguishing it from a real
                // call's own resumption `Fix` below, structurally, by
                // *which* label `App` targets: itself (a loop) vs. some
                // other real function (a call).
                CExpr::App {
                    func: CVal::Label(callee),
                    args,
                } if callee == &def.name => {
                    lower_loop(
                        ctx,
                        block,
                        env,
                        k_ret,
                        result_type,
                        yield_targets,
                        def,
                        args,
                    );
                }
                CExpr::App {
                    func: CVal::Label(callee),
                    args,
                } if matches!(args.last(), Some(CVal::Label(l)) if l == &def.name) => {
                    lower_real_call(
                        ctx,
                        block,
                        env,
                        k_ret,
                        result_type,
                        yield_targets,
                        def,
                        callee,
                        args,
                    );
                }
                _ => panic!(
                    "MLIR lowering doesn't support this `Fix` shape yet -- see mlir_lower.rs's own module doc comment"
                ),
            }
        }
        _ => panic!(
            "MLIR lowering doesn't support this CPS shape yet (return, extern/intrinsic LetPrim chains, if-with-result, real calls, and loops, so far) -- see mlir_lower.rs's own module doc comment"
        ),
    }
}

/// A two-armed `if` used as an expression (`Fix{ defs: [join], body: If{..}
/// }`, `cps.rs`'s own shape for `ExprKind::If`) lowers to `scf.if`: both
/// arms are lowered into their own fresh, argument-less block with
/// `yield_label` set to `join`'s own name, so each arm's tail call to it
/// (`tail_call_join` in `cps.rs`) becomes `scf.yield` instead. The `scf.if`
/// operation's own results are then bound to the join's own params in
/// order, and lowering continues into the join's own body (ordinary flow,
/// `yield_label` reset to whatever it was *before* this `if` — the same one
/// this call itself received, so a nested `if` inside a bigger one still
/// resolves its *own* join correctly).
///
/// `join.params` is `[result_var, ...carried]` — the `if`'s own value,
/// followed by one entry per outer variable either branch reassigns
/// (`cps.rs`'s own `ExprKind::If` handling, "Mutation across control flow")
/// — and `join.carried_types` (populated there too) carries each position's
/// own real type in the same order, exactly the mechanism `lower_loop`
/// already established for its own carried state (see `CFunDef::
/// carried_types`'s own doc comment for why guessing one shared type here
/// was wrong): an `if` nested inside a loop, whose own branches reassign
/// some outer flag, is exactly the case that needed this — confirmed
/// broken before this generalization, found by direct testing.
fn lower_if<'c>(
    ctx: &LowerCtx<'c, '_>,
    block: &Block<'c>,
    env: HashMap<CVar, Value<'c, 'c>>,
    k_ret: CVar,
    result_type: Type<'c>,
    yield_targets: &[YieldTarget<'c, '_>],
    join: &CFunDef,
    cond: &CVal,
    then_branch: &CExpr,
    else_branch: &CExpr,
) {
    let Some(join_cleave_types) = &join.carried_types else {
        panic!(
            "MLIR lowering: an `if`-join's own `CFunDef` must carry `carried_types` -- see mlir_lower.rs's own module doc comment"
        );
    };
    // A unit-typed position (most commonly `join.params[0]`, the `if`'s own
    // value, for a bodyless-`else`/statement-position `if`) carries no real
    // MLIR value at all -- `ty_to_mlir` has none to give it, and there's
    // nothing meaningful to yield/carry. `live` is `join.params`' own
    // indices with a real (non-unit) type, in order -- both the eventual
    // `scf.if` result list and `join.params`-to-result binding below walk
    // this instead of `join.params` directly, keeping every position
    // consistent with the *other* `CVal::Unit`-filtering already needed on
    // the `scf.yield` side (see that arm's own doc comment).
    let live: Vec<usize> = (0..join_cleave_types.len())
        .filter(|&i| !is_unit_ty(&join_cleave_types[i]))
        .collect();
    let join_types: Vec<Type> = live
        .iter()
        .map(|&i| ty_to_mlir(ctx, &join_cleave_types[i]))
        .collect();

    let context = ctx.context;
    let location = Location::unknown(context);
    let bool_ty: Type = IntegerType::new(context, 1).into();
    let cond_value = lower_cval(context, block, &env, cond, bool_ty);

    // Pushed onto a *fresh* `Vec` for the two branches below — the original
    // `yield_targets` (without this entry) is what the join's own
    // continuation, after the `scf.if` is built, uses instead (see
    // `lower_cexpr`'s own doc comment for why this needs to be a stack, not
    // a single replaced slot).
    let mut inner_targets: Vec<YieldTarget<'c, '_>> = yield_targets.to_vec();
    // `None` -- an `if`-join is never an iteration boundary, see `YieldTarget`'s own doc comment.
    inner_targets.push((&join.name, &join_types, None));

    let then_block = Block::new(&[]);
    lower_cexpr(
        ctx,
        &then_block,
        env.clone(),
        k_ret,
        result_type,
        &inner_targets,
        then_branch,
    );
    let then_region = Region::new();
    then_region.append_block(then_block);

    let else_block = Block::new(&[]);
    lower_cexpr(
        ctx,
        &else_block,
        env.clone(),
        k_ret,
        result_type,
        &inner_targets,
        else_branch,
    );
    let else_region = Region::new();
    else_region.append_block(else_block);

    let if_op = block.append_operation(scf::r#if(
        cond_value,
        &join_types,
        then_region,
        else_region,
        location,
    ));

    // Only `live` positions got a real `scf.if` result at all -- a unit
    // position's own `join.params` entry is left unbound, matching `PrimOp::
    // Store`'s own "bound result is unit and never read" convention: nothing
    // downstream ever actually materializes it as a value (a discarded
    // statement's own continuation ignores what it's handed), only panics
    // clearly (`lower_cval`'s own "unbound CPS variable") in the genuine
    // edge case where something unexpectedly tries to.
    let mut env = env;
    for (result_idx, &param_idx) in live.iter().enumerate() {
        env.insert(
            join.params[param_idx],
            if_op.result(result_idx).unwrap().into(),
        );
    }
    lower_cexpr(
        ctx,
        block,
        env,
        k_ret,
        result_type,
        yield_targets,
        &join.body,
    );
}

/// A `while`/`for` loop (CPS unifies both into the same shape — a self-
/// recursive continuation carrying loop state, see `doc/backlog.md`'s own
/// "Stage 3" note) lowers to `scf.while`. Recognized in `lower_cexpr`'s own
/// `Fix` arm as `Fix{ defs: [loop_def], body: App{Label(name), initial_args}
/// }` where `name == loop_def.name` — the loop's own *entry*, distinguished
/// structurally from a real call's own resumption `Fix` by *which* label
/// the `App` targets (itself, vs. some other real function).
///
/// `loop_def`'s own body must itself be `Fix{ defs: [cond_k], body: App{
/// Label(real_fn), cond_args} }` — the condition, always a real call
/// (`Ord::lt<...>` etc., since comparisons are real functions now, not
/// straight-line intrinsics) — whose own continuation `cond_k`'s body is a
/// *bare* `If` (no synthesized join needed: `cps.rs`'s own `ExprKind::
/// While`/`For` doc comment notes both arms already terminate on their own,
/// unlike an `if` used as an expression). That real call is inlined here by
/// hand (not via `lower_real_call`, which also recurses into ordinary
/// `lower_cexpr` afterward — the `if` that follows needs special handling
/// as the region boundary itself, not as ordinary control flow).
///
/// The `then` branch's own tail recursion back to `loop_def.name` reuses
/// `lower_cexpr`'s *existing* `yield_targets`-checking `App` arm — pushing
/// `(&loop_def.name, &carried_types)` onto the stack while lowering it turns
/// that recursive tail-call into `scf.yield` for free, no new recognition
/// needed (also what lets a `break` nested inside an `if` inside this loop's
/// own body find `loop_def.name` past the `if`-join sitting between them —
/// see `lower_cexpr`'s own doc comment). The `else` branch (loop exit) is
/// lowered *after* the `scf.while` op is built, in the *outer* scope —
/// `loop_def`'s own params bound to the op's own results, the outer
/// `k_ret`/`yield_targets` unchanged (mirrors `lower_if`'s own "join's
/// continuation runs in ordinary flow" step).
fn lower_loop<'c>(
    ctx: &LowerCtx<'c, '_>,
    block: &Block<'c>,
    env: HashMap<CVar, Value<'c, 'c>>,
    k_ret: CVar,
    result_type: Type<'c>,
    yield_targets: &[YieldTarget<'c, '_>],
    loop_def: &CFunDef,
    initial_args: &[CVal],
) {
    let context = ctx.context;
    let location = Location::unknown(context);

    // A loop's own condition is *zero or more* sequential real calls, not
    // always exactly one — `i < hull.len()` needs two (`DynArray::len<...>`,
    // then `Ord::lt<i32>` against its result), found directly while writing
    // `examples/convex_hull.cleave` (`doc/backlog-done.md`'s own "a while-
    // loop condition needing more than one chained real call" item); a bare
    // `loop { }` needs *zero* — its own condition is just the synthetic
    // `running` flag, read directly, no call at all (`doc/backlog-done.md`'s
    // own "break value" item). Walked here as a chain of `Fix{defs:[k],
    // body: App{Label(callee), args}}` layers, each `k`'s own body either
    // another such layer or, terminally, the bare `If` — mirrors the exact
    // same "keep following the next `Fix`" shape ordinary straight-line
    // lowering already uses (`lower_real_call`'s own recursive continuation-
    // following), just walked by hand in a loop here since each call in the
    // chain has to land in the "before" block below, ending in `scf.
    // condition`, not an ordinary recursive `lower_cexpr` step.
    struct CondCall<'e> {
        callee: &'e str,
        args: &'e [CVal],
        result_var: CVar,
    }
    let mut cond_calls: Vec<CondCall> = Vec::new();
    let mut cursor: &CExpr = &loop_def.body;
    let (cond, then_branch, else_branch) = loop {
        match cursor {
            CExpr::If {
                cond,
                then_branch,
                else_branch,
            } => break (cond, then_branch, else_branch),
            CExpr::Fix {
                defs: cond_defs,
                body: cond_body,
            } => {
                let [cond_k] = &cond_defs[..] else {
                    panic!("MLIR lowering doesn't support a multi-def loop condition yet");
                };
                let CExpr::App {
                    func: CVal::Label(cond_callee),
                    args: cond_args,
                } = &**cond_body
                else {
                    panic!(
                        "MLIR lowering: a loop's own condition must be a real call -- see mlir_lower.rs's own module doc comment"
                    );
                };
                let [cond_result_var] = cond_k.params[..] else {
                    panic!(
                        "MLIR lowering: a loop's own condition call must have exactly one result"
                    );
                };
                cond_calls.push(CondCall {
                    callee: cond_callee,
                    args: cond_args,
                    result_var: cond_result_var,
                });
                cursor = &cond_k.body;
            }
            _ => panic!(
                "MLIR lowering: a loop's own condition must be a chain of real calls ending in a bare `if` (or a bare `if` directly) -- see mlir_lower.rs's own module doc comment"
            ),
        }
    };

    // Carried-state types, from `loop_def.carried_types` (`cps.rs`'s own
    // `ExprKind::While`/`For` conversion, one `Ty` per `loop_def.params`
    // position) -- *not* derived from `init_values` themselves: a carried
    // variable's own initial value can be a bare, width-less literal CVal
    // (`total = 0.0`, never wrapped in a `LetPrim`), and using one shared
    // `result_type` as `lower_cval`'s own `expected_type` for *every*
    // initial value (the previous approach) is wrong the moment a loop
    // carries more than one type, or a type other than the enclosing
    // function's own result — found by direct testing (see `CFunDef::
    // carried_types`'s own doc comment for the exact failure this caused).
    let Some(carried_cleave_types) = &loop_def.carried_types else {
        panic!(
            "MLIR lowering: a loop's own `CFunDef` must carry `carried_types` -- see mlir_lower.rs's own module doc comment"
        );
    };
    let carried_types: Vec<Type> = carried_cleave_types
        .iter()
        .map(|t| ty_to_mlir(ctx, t))
        .collect();
    let init_values: Vec<Value> = initial_args
        .iter()
        .zip(&carried_types)
        .map(|(a, &t)| lower_cval(context, block, &env, a, t))
        .collect();

    // -- "before" region: receives the carried state, checks the
    // condition, and either continues (`scf.condition` true) into "after"
    // or exits the whole `scf.while` op (false) --
    let before_block = Block::new(
        &carried_types
            .iter()
            .map(|&t| (t, location))
            .collect::<Vec<_>>(),
    );
    // Starts from a *copy* of the outer `env`, not empty — a value bound
    // outside the loop (an enclosing function's own parameter, an outer
    // `let`, ...) still dominates this nested region's own blocks in MLIR,
    // and the loop's own condition/body can reference it freely (only a
    // *mutated* variable needs to be threaded as explicit carried state —
    // see `cps.rs`'s own "Mutation across control flow" doc comment; an
    // untouched outer reference doesn't). Found by direct testing: an empty
    // `before_env`/`after_env` here previously produced a clean "unbound CPS
    // variable" panic the moment a loop body referenced *any* outer,
    // non-carried variable — a real, pre-existing gap, never exercised
    // before a loop body needed a struct field reached through an outer
    // (function-parameter-level) struct reference.
    let mut before_env: HashMap<CVar, Value> = env.clone();
    for (i, &p) in loop_def.params.iter().enumerate() {
        before_env.insert(p, before_block.argument(i).unwrap().into());
    }
    // Emit every call in the condition's own chain, in order (empty for a
    // bare `loop { }` — see this function's own doc comment above), each one
    // seeing the previous ones' results already bound in `before_env` (only
    // relevant for arguments referencing an earlier call's own result, e.g.
    // `Ord::lt`'s second argument here referencing `hull.len()`'s).
    for cond_call in &cond_calls {
        let Some((cond_param_types, cond_result_ty)) = ctx.signatures.get(cond_call.callee) else {
            panic!(
                "MLIR lowering: call to unknown top-level fn `{}` in a loop condition",
                cond_call.callee
            );
        };
        let cond_result_mlir_ty = ty_to_mlir(ctx, cond_result_ty);
        // `args`' own last entry is the synthesized continuation label
        // itself (`emit_call`'s own convention, see `cps.rs`), not a real arg.
        let real_cond_args = &cond_call.args[..cond_call.args.len() - 1];
        let cond_arg_values: Vec<Value> = real_cond_args
            .iter()
            .zip(cond_param_types)
            .map(|(a, t)| lower_cval(context, &before_block, &before_env, a, ty_to_mlir(ctx, t)))
            .collect();
        let cond_call_op = before_block.append_operation(func::call(
            context,
            FlatSymbolRefAttribute::new(context, cond_call.callee),
            &cond_arg_values,
            &[cond_result_mlir_ty],
            location,
        ));
        let result: Value = cond_call_op.result(0).unwrap().into();
        before_env.insert(cond_call.result_var, result);
    }
    // The terminal `If`'s own `cond` — when `cond_calls` is non-empty,
    // this is a bare `CVal::Var` naming the *last* call's own result,
    // already bound in `before_env` just above; when empty (a bare `loop`,
    // whose own "condition" is just the synthetic `running` flag, read
    // directly, no call needed at all), it resolves straight through
    // `before_env`'s own pre-existing carried-param bindings instead. One
    // `lower_cval` call handles both uniformly.
    let bool_ty: Type = IntegerType::new(context, 1).into();
    let cond_result = lower_cval(context, &before_block, &before_env, cond, bool_ty);
    let carried_before: Vec<Value> = loop_def.params.iter().map(|p| before_env[p]).collect();
    before_block.append_operation(scf::condition(cond_result, &carried_before, location));
    let before_region = Region::new();
    before_region.append_block(before_block);

    // -- "after" region: the loop's own real body, `then_branch`'s own
    // tail recursion to `loop_def.name` becoming `scf.yield` --
    let after_block = Block::new(
        &carried_types
            .iter()
            .map(|&t| (t, location))
            .collect::<Vec<_>>(),
    );
    // See `before_env`'s own doc comment above — same fix, same reason.
    let mut after_env: HashMap<CVar, Value> = env.clone();
    for (i, &p) in loop_def.params.iter().enumerate() {
        after_env.insert(p, after_block.argument(i).unwrap().into());
    }

    // `doc/hld.md`'s own "Memory management" section, the arena's first
    // real application (`region_analysis.rs`'s own module doc comment has
    // the full reasoning): one region per loop *iteration*, opened here at
    // the very start of the body, closed by `lower_cexpr`'s own `App`-to-
    // `yield_targets` arm right before this same iteration's own tail-call
    // (`YieldTarget`'s own doc comment) — spanning the *whole* iteration,
    // not just an individual call within it, because a region-local
    // function's own result can genuinely need to stay valid past its own
    // call returning (`net_grad`'s own `g.2`, read afterward by `Optimizer
    // ::step`, is exactly this shape) — safe to open unconditionally, even
    // around calls that are *not* region-local (`Optimizer::step`'s own
    // allocation sites never call `cleave_alloc_local` at all, regardless
    // of whether a region happens to be open around their execution — the
    // decision was already made once, per allocation *site*, at compile
    // time, `alloc_llvm_value`'s own doc comment).
    let i64_ty: Type = IntegerType::new(context, 64).into();
    ensure_extern_declared(
        ctx,
        "cleave_region_enter",
        &[Ty::Con("i64".to_string())],
        &[i64_ty],
    );
    let zero_size = after_block
        .append_operation(arith::constant(
            context,
            IntegerAttribute::new(i64_ty, 0).into(),
            location,
        ))
        .result(0)
        .unwrap()
        .into();
    let region_handle: Value = after_block
        .append_operation(func::call(
            context,
            FlatSymbolRefAttribute::new(context, "cleave_region_enter"),
            &[zero_size],
            &[i64_ty],
            location,
        ))
        .result(0)
        .unwrap()
        .into();

    // Pushed onto a *fresh* `Vec` for the body — the original `yield_targets`
    // (without this entry) is what the loop-exit continuation, after the
    // `scf.while` is built, uses instead (see `lower_cexpr`'s own doc
    // comment for why this needs to be a stack, not a single replaced slot).
    let mut inner_targets: Vec<YieldTarget<'c, '_>> = yield_targets.to_vec();
    inner_targets.push((&loop_def.name, &carried_types, Some(region_handle)));
    lower_cexpr(
        ctx,
        &after_block,
        after_env,
        k_ret,
        result_type,
        &inner_targets,
        then_branch,
    );
    let after_region = Region::new();
    after_region.append_block(after_block);

    let while_op = block.append_operation(scf::r#while(
        &init_values,
        &carried_types,
        before_region,
        after_region,
        location,
    ));

    // The `else` branch (loop exit) runs in the *outer* scope, ordinary
    // flow -- `loop_def`'s own params now bound to `scf.while`'s own
    // results (its carried state as of the moment the condition failed).
    let mut env = env;
    for (i, &p) in loop_def.params.iter().enumerate() {
        env.insert(p, while_op.result(i).unwrap().into());
    }
    lower_cexpr(
        ctx,
        block,
        env,
        k_ret,
        result_type,
        yield_targets,
        else_branch,
    );
}

/// A real call to another top-level cleave `fn` (`emit_call`'s own
/// `UnitBody::Real` shape in `cps.rs`: `Fix{ defs: [k], body: App{
/// Label(callee), [...args, Label(k.name)] } }`) lowers to an ordinary
/// `func.call` -- unlike an `extern` call there's no declaration to emit
/// (the callee is another `func.func` `lower_program` already builds into
/// this same module, symbol resolution across the module doesn't care about
/// textual order, including a function calling itself). The synthesized
/// continuation `k` is always exactly one parameter (`emit_call` never
/// builds it any other way), bound to the call's own result, with lowering
/// continuing straight into `k`'s own body in the same block -- synchronous
/// control flow, no actual closure needed.
fn lower_real_call<'c>(
    ctx: &LowerCtx<'c, '_>,
    block: &Block<'c>,
    env: HashMap<CVar, Value<'c, 'c>>,
    k_ret: CVar,
    result_type: Type<'c>,
    yield_targets: &[YieldTarget<'c, '_>],
    k: &CFunDef,
    callee: &str,
    args: &[CVal],
) {
    let [result_var] = k.params[..] else {
        panic!(
            "MLIR lowering: a real call's own synthesized continuation must have exactly one parameter"
        );
    };
    let Some((param_types, result_ty)) = ctx.signatures.get(callee) else {
        panic!("MLIR lowering: call to unknown top-level fn `{callee}`");
    };
    let context = ctx.context;
    // `args`' own last entry is the synthesized continuation label itself
    // (`emit_call`'s own convention, see `cps.rs`), not a real argument.
    let real_args = &args[..args.len() - 1];
    let arg_values: Vec<Value> = real_args
        .iter()
        .zip(param_types)
        .map(|(a, t)| lower_cval(context, block, &env, a, ty_to_mlir(ctx, t)))
        .collect();
    let location = Location::unknown(context);
    // A `()`-returning callee is declared with *zero* MLIR results
    // (`lower_top_level_fn`'s own `is_unit`/`results` handling, applied to
    // every top-level fn, not just `main`) -- the call site must match that
    // exactly, or MLIR's own verifier rejects it ("incorrect number of
    // results for callee"), found by direct testing (a real call to a
    // `()`-returning function used to unconditionally request one result
    // here regardless of `result_ty`). `emit_call` (`cps.rs`) always
    // synthesizes a `result_var` regardless of the callee's own return type
    // -- unlike `PrimOp::Store`'s dedicated "bound result is unit, never
    // read" convention (`lower_prim_op` returns `None`), a real call has no
    // such per-callee special case at the CPS level, so it's handled here
    // instead: `result_var` simply never gets bound into `env` when unit,
    // matching every other place in this file where a unit-typed value is
    // never materialized (nothing in the language can do anything with one
    // besides discard it, so nothing downstream ever looks it up).
    let is_unit = is_unit_ty(result_ty);
    let results: Vec<Type> = if is_unit {
        vec![]
    } else {
        vec![ty_to_mlir(ctx, result_ty)]
    };
    let call_op = block.append_operation(func::call(
        context,
        FlatSymbolRefAttribute::new(context, callee),
        &arg_values,
        &results,
        location,
    ));

    let mut env = env;
    if !is_unit {
        env.insert(result_var, call_op.result(0).unwrap().into());
    }
    lower_cexpr(ctx, block, env, k_ret, result_type, yield_targets, &k.body);
}

/// Returns `None` exactly for `PrimOp::Store` — a real effect with zero MLIR
/// results (`memref.store`), matching `cps.rs`'s own documentation that a
/// `Store`'s bound result "is unit and never read": `lower_cexpr`'s own
/// `LetPrim` arm skips the `env.insert` entirely in that case, rather than
/// fabricating a placeholder value nothing should ever look up.
/// `ty` is the `LetPrim`'s own declared result type — resolved to an MLIR
/// `Type` (`ty_to_mlir`) only by the arms that actually need one; `Store`'s
/// own `ty` is always the unit type `()` (see this function's own doc
/// comment above), which `ty_to_mlir` doesn't support at all, so it must
/// never be converted unconditionally here.
fn lower_prim_op<'c>(
    ctx: &LowerCtx<'c, '_>,
    block: &Block<'c>,
    env: &HashMap<CVar, Value<'c, 'c>>,
    op: &PrimOp,
    args: &[CVal],
    ty: &Ty,
) -> Option<Value<'c, 'c>> {
    match op {
        PrimOp::Extern {
            symbol,
            param_types,
        } => {
            // A `Ty::Array`-typed *return* is a real ABI mismatch to
            // reconcile: the cleave-level call (e.g. `Print<[i8;N]>::
            // print(x) -> x`, an identity contract every other `Print<T>`
            // impl already has) says `[i8;N]`, but the real C symbol it's
            // backed by (`print_bytes`) returns a plain scalar -- nothing
            // in `cleave-rt` could plausibly hand back a `memref` anyway.
            // The extern symbol's own *declared* return becomes `i64`
            // (an arbitrary-but-consistent small-scalar convention for
            // "the real C fn returns some byte count/status, discarded"),
            // and the cleave-level result reuses whichever argument was
            // already that same array type -- the identity case this
            // module's own single real use case needs, not a general
            // "reconstruct an array from a scalar" mechanism (impossible
            // anyway).
            // A `()`-returning extern (`extern fn touch(x: i32);`, no `->`
            // clause) needs the exact same zero-results treatment `lower_
            // real_call`/`lower_top_level_fn` already give a unit-returning
            // top-level fn -- `ty_to_mlir` has no case for `()` at all (it
            // falls through to the generic-struct arm, `!llvm.ptr`), which
            // used to silently declare/call against a bogus pointer result
            // matching nothing the real C symbol underneath actually
            // returns (found by direct testing: a real, structural ABI
            // mismatch, though not one that happened to crash on its own —
            // nothing downstream ever read the bogus value).
            let is_array_return = matches!(ty, Ty::Array(..));
            let is_unit_return = is_unit_ty(ty);
            let results: Vec<Type> = if is_array_return {
                vec![IntegerType::new(ctx.context, 64).into()]
            } else if is_unit_return {
                vec![]
            } else {
                vec![ty_to_mlir(ctx, ty)]
            };
            ensure_extern_declared(ctx, symbol, param_types, &results);
            // A `Ty::Array`-typed argument crosses the call boundary as a
            // raw pointer + a compile-time-known length (two scalars),
            // never as its own `memref` value directly -- see
            // `array_ptr_and_len`'s own doc comment.
            let mut arg_values: Vec<Value> = Vec::new();
            let mut array_identity: Option<Value> = None;
            for (a, t) in args.iter().zip(param_types) {
                let lowered = lower_cval(ctx.context, block, env, a, ty_to_mlir(ctx, t));
                if matches!(t, Ty::Array(..)) {
                    if is_array_return && array_identity.is_none() {
                        array_identity = Some(lowered);
                    }
                    let (ptr, len) = array_ptr_and_len(ctx, block, lowered, t);
                    arg_values.push(ptr);
                    arg_values.push(len);
                } else {
                    arg_values.push(lowered);
                }
            }
            let location = Location::unknown(ctx.context);
            let call_op = block.append_operation(func::call(
                ctx.context,
                FlatSymbolRefAttribute::new(ctx.context, symbol),
                &arg_values,
                &results,
                location,
            ));
            if is_array_return {
                Some(array_identity.unwrap_or_else(|| panic!("MLIR lowering: extern `{symbol}` declares an array return but has no array-typed argument to reuse as its identity")))
            } else if is_unit_return {
                None
            } else {
                Some(call_op.result(0).unwrap().into())
            }
        }
        PrimOp::RawMlirOp { op, attrs } => Some(lower_raw_mlir_op(
            ctx,
            block,
            env,
            op,
            attrs,
            args,
            ty_to_mlir(ctx, ty),
        )),
        PrimOp::Array => Some(lower_array_construct(ctx, block, env, ty, args)),
        PrimOp::ArrayRepeat => Some(lower_array_repeat(ctx, block, env, ty, args)),
        PrimOp::Load { array_ty } => Some(lower_array_load(ctx, block, env, array_ty, args)),
        PrimOp::Store { array_ty } => {
            lower_array_store(ctx, block, env, array_ty, args);
            None
        }
        PrimOp::Struct(name, field_names) => {
            if native_shape_keyword(ctx, name).is_some() {
                Some(lower_tagged_struct_construct(ctx, block, env, ty, args))
            } else {
                Some(lower_struct_construct(
                    ctx,
                    block,
                    env,
                    ty,
                    field_names,
                    args,
                ))
            }
        }
        PrimOp::Field { struct_ty, field } => {
            Some(lower_field_access(ctx, block, env, struct_ty, field, args))
        }
        PrimOp::FieldStore { struct_ty, field } => {
            lower_field_store(ctx, block, env, struct_ty, field, args);
            None
        }
        PrimOp::Retain(rc_ty) => {
            lower_refcount_call(ctx, block, env, "cleave_retain", rc_ty, args);
            None
        }
        PrimOp::Release(rc_ty) => {
            let CVal::Var(ptr_var) = &args[0] else {
                panic!("MLIR lowering: `cleave_release`'s own operand must be a variable");
            };
            let ptr_val = *env
                .get(ptr_var)
                .unwrap_or_else(|| panic!("MLIR lowering: unbound CPS variable v{ptr_var}"));
            lower_release_cascade(ctx, block, rc_ty, ptr_val);
            None
        }
    }
}

/// `ExprKind::ArrayLit`'s own CPS conversion (`cps.rs`) is fully generic over
/// each element `Expr` — a *nested* literal (`[[1,2],[3,4]]`) therefore
/// produces its inner rows as their own, separately-lowered `PrimOp::Array`
/// values first, with the *outer* `Array`'s own `args` referencing them, not
/// scalars. Since this file flattens the whole nested `Ty::Array` chain into
/// *one* memref (`array_memref_type`), never memref-of-memrefs, such an arg
/// is copied elementwise into the right slice of the outer memref
/// (`copy_nested_array`) rather than stored as a reference; an arg is
/// scalar exactly when `ty`'s own flattened shape is one dimension deep for
/// that position — a property of the *type*, not of runtime inspection.
///
/// A struct-typed leaf (`[Point; N]`) takes a wholly different path
/// (`array_leaf_is_struct`): a `memref` can't hold a struct's own opaque
/// `!llvm.ptr` element type (`MemRefType::new` rejects it with a hard native
/// assertion, found by direct testing) — instead a real heap allocation
/// (`alloc_llvm_value`, shaped as the inline `!llvm.array<N x !llvm.ptr>`
/// `ty_to_llvm_field_type` already builds for a struct's own array-typed
/// field) is filled one `llvm.getelementptr`+`llvm.store` per element,
/// mirroring `lower_struct_construct`'s own per-field GEP+store loop. Only a
/// single-dimension struct-leaf array is handled — a struct-leaf array
/// nested inside another array is real but rarer, and panics clearly here
/// rather than being silently mishandled.
fn lower_array_construct<'c>(
    ctx: &LowerCtx<'c, '_>,
    block: &Block<'c>,
    env: &HashMap<CVar, Value<'c, 'c>>,
    ty: &Ty,
    args: &[CVal],
) -> Value<'c, 'c> {
    let (dims, leaf_ty) = flatten_array_dims(ty);
    let Some((_, inner_dims)) = dims.split_first() else {
        panic!("MLIR lowering: `array` prim on a non-array type `{ty}`");
    };
    if array_leaf_is_struct(ctx, ty) {
        assert!(
            inner_dims.is_empty(),
            "MLIR lowering: a struct-leaf array nested inside another array (`{ty}`) isn't supported yet -- only a single-dimension struct-leaf array (`[Struct; N]`) is"
        );
        let array_llvm_ty = ty_to_llvm_field_type(ctx, ty);
        let elem_ty = ty_to_mlir(ctx, leaf_ty);
        let ptr = alloc_llvm_value(ctx, block, array_llvm_ty);
        let location = Location::unknown(ctx.context);
        for (i, arg) in args.iter().enumerate() {
            let elem_val = lower_cval(ctx.context, block, env, arg, elem_ty);
            let dst_ptr = gep(ctx, block, ptr, &[0, i as i64], array_llvm_ty);
            block.append_operation(llvm::store(
                ctx.context,
                elem_val,
                dst_ptr,
                location,
                LoadStoreOptions::new(),
            ));
        }
        return ptr;
    }
    let array_val = alloc_array(
        ctx,
        block,
        MemRefType::new(ty_to_mlir(ctx, leaf_ty), &dims, None, None),
    );
    if inner_dims.is_empty() {
        let elem_ty = ty_to_mlir(ctx, leaf_ty);
        let location = Location::unknown(ctx.context);
        for (i, arg) in args.iter().enumerate() {
            let elem_val = lower_cval(ctx.context, block, env, arg, elem_ty);
            let idx = const_index(ctx, block, i as i64);
            block.append_operation(memref::store(elem_val, array_val, &[idx], location));
        }
    } else {
        for (i, arg) in args.iter().enumerate() {
            let src = lower_nested_array_arg(env, arg);
            let idx = const_index(ctx, block, i as i64);
            copy_nested_array(ctx, block, src, inner_dims, array_val, &[idx]);
        }
    }
    array_val
}

/// `[value; N]` — `N` is *not* read from `args[1]` at all: `ty`'s own
/// flattened outer dimension is already the authoritative, type-checked
/// count (the same invariant `PrimOp::Array`'s own `args.len()` relies on),
/// simpler than re-materializing a separate compile-time-constant CVal for a
/// value this function already has from `ty`.
fn lower_array_repeat<'c>(
    ctx: &LowerCtx<'c, '_>,
    block: &Block<'c>,
    env: &HashMap<CVar, Value<'c, 'c>>,
    ty: &Ty,
    args: &[CVal],
) -> Value<'c, 'c> {
    let (dims, leaf_ty) = flatten_array_dims(ty);
    let Some((&outer_dim, inner_dims)) = dims.split_first() else {
        panic!("MLIR lowering: `array-repeat` prim on a non-array type `{ty}`");
    };
    let value_arg = &args[0];
    if array_leaf_is_struct(ctx, ty) {
        assert!(
            inner_dims.is_empty(),
            "MLIR lowering: a struct-leaf array nested inside another array (`{ty}`) isn't supported yet -- only a single-dimension struct-leaf array (`[Struct; N]`) is"
        );
        let array_llvm_ty = ty_to_llvm_field_type(ctx, ty);
        let elem_ty = ty_to_mlir(ctx, leaf_ty);
        let ptr = alloc_llvm_value(ctx, block, array_llvm_ty);
        let elem_val = lower_cval(ctx.context, block, env, value_arg, elem_ty);
        let location = Location::unknown(ctx.context);
        for i in 0..outer_dim {
            let dst_ptr = gep(ctx, block, ptr, &[0, i], array_llvm_ty);
            block.append_operation(llvm::store(
                ctx.context,
                elem_val,
                dst_ptr,
                location,
                LoadStoreOptions::new(),
            ));
        }
        return ptr;
    }
    let array_val = alloc_array(
        ctx,
        block,
        MemRefType::new(ty_to_mlir(ctx, leaf_ty), &dims, None, None),
    );
    if inner_dims.is_empty() {
        let elem_ty = ty_to_mlir(ctx, leaf_ty);
        let elem_val = lower_cval(ctx.context, block, env, value_arg, elem_ty);
        let location = Location::unknown(ctx.context);
        for i in 0..outer_dim {
            let idx = const_index(ctx, block, i);
            block.append_operation(memref::store(elem_val, array_val, &[idx], location));
        }
    } else {
        let src = lower_nested_array_arg(env, value_arg);
        for i in 0..outer_dim {
            let idx = const_index(ctx, block, i);
            copy_nested_array(ctx, block, src, inner_dims, array_val, &[idx]);
        }
    }
    array_val
}

/// A nested array literal/repeat's own element is always an already-built
/// array value bound to a CPS variable — never a bare literal (arrays have
/// no literal `CVal` form) — so `expected_type` (as `lower_cval` otherwise
/// needs for a bare literal) never applies here.
fn lower_nested_array_arg<'c>(env: &HashMap<CVar, Value<'c, 'c>>, arg: &CVal) -> Value<'c, 'c> {
    let CVal::Var(v) = arg else {
        panic!(
            "MLIR lowering: a nested array's own element must be an already-built array value, not a bare literal"
        )
    };
    *env.get(v)
        .unwrap_or_else(|| panic!("MLIR lowering: unbound CPS variable v{v}"))
}

/// `args = [array, index...]`, `array_ty` the array's own concrete cleave
/// type (`PrimOp::Load`'s own payload, see its doc comment for why).
/// `collect_index_chain` (`cps.rs`) has already collapsed a whole `a[i,j]`
/// chain into one multi-index op by this stage. Two representations, picked
/// by checking `array_val`'s own already-lowered MLIR type: a **standalone**
/// array is a self-describing `memref` (`memref.load`, indices bridged from
/// ordinary `i32` to `index` via `arith.index_cast` — MLIR's own
/// `memref.load`/`store` require `index`-typed operands specifically); a
/// **struct-embedded** array (reached through a `PrimOp::Field` first — see
/// `lower_field_access`'s own doc comment) is an opaque `!llvm.ptr` already
/// positioned at the start of its own (possibly nested) `!llvm.array`, with
/// no shape of its own to query — `array_ty` supplies it instead, walked via
/// one combined `llvm.getelementptr` (leading `0` index to stay within this
/// one array instance, matching `lower_struct_construct`'s own field-GEP
/// convention) plus `llvm.load`.
fn lower_array_load<'c>(
    ctx: &LowerCtx<'c, '_>,
    block: &Block<'c>,
    env: &HashMap<CVar, Value<'c, 'c>>,
    array_ty: &Ty,
    args: &[CVal],
) -> Value<'c, 'c> {
    let CVal::Var(array_var) = &args[0] else {
        panic!("MLIR lowering: `load`'s own array operand must be a variable");
    };
    let array_val = *env
        .get(array_var)
        .unwrap_or_else(|| panic!("MLIR lowering: unbound CPS variable v{array_var}"));
    let i32_ty = width_ty(ctx, "i32");
    let location = Location::unknown(ctx.context);
    if array_val.r#type().is_mem_ref() {
        let index_vals: Vec<Value> = args[1..]
            .iter()
            .map(|a| to_index(ctx, block, lower_cval(ctx.context, block, env, a, i32_ty)))
            .collect();
        let load_op = block.append_operation(memref::load(array_val, &index_vals, location));
        load_op.result(0).unwrap().into()
    } else if array_val.r#type().is_tensor() {
        // A `#[mlir_type(tensor)]`-tagged struct's own field, read back
        // through `lower_field_access`'s identity path — a bare native
        // `tensor<...>` SSA value, never a memref or `!llvm.ptr`. `tensor.
        // extract` is the read-only, purely functional equivalent of
        // `memref.load` for this representation.
        let (_, leaf_ty) = flatten_array_dims(array_ty);
        let index_vals: Vec<Value> = args[1..]
            .iter()
            .map(|a| to_index(ctx, block, lower_cval(ctx.context, block, env, a, i32_ty)))
            .collect();
        let result_ty = ty_to_mlir(ctx, leaf_ty);
        let built = OperationBuilder::new("tensor.extract", location)
            .add_operands(&[array_val])
            .add_operands(&index_vals)
            .add_results(&[result_ty])
            .build()
            .unwrap_or_else(|e| panic!("MLIR lowering: failed to build tensor.extract: {e}"));
        block.append_operation(built).result(0).unwrap().into()
    } else {
        let (_, leaf_ty) = flatten_array_dims(array_ty);
        let array_llvm_ty = ty_to_llvm_field_type(ctx, array_ty);
        let mut gep_indices = vec![const_i32(ctx, block, 0)];
        gep_indices.extend(
            args[1..]
                .iter()
                .map(|a| lower_cval(ctx.context, block, env, a, i32_ty)),
        );
        let leaf_ptr = gep_dynamic(ctx, block, array_val, &gep_indices, array_llvm_ty);
        let result_ty = ty_to_mlir(ctx, leaf_ty);
        block
            .append_operation(llvm::load(
                ctx.context,
                leaf_ptr,
                result_ty,
                location,
                LoadStoreOptions::new(),
            ))
            .result(0)
            .unwrap()
            .into()
    }
}

/// `args = [array, index..., value]` (value last — `cps.rs`'s own
/// `StmtKind::Assign`'s `Index` arm convention), `array_ty` — see
/// `lower_array_load`'s own doc comment (same two representations, same
/// dispatch). The value's own expected type comes from the array's own
/// element type (`MemRefType::try_from`+`.element()` for the memref case,
/// `array_ty`'s own flattened leaf for the pointer case) — not from `ty`, a
/// `Store`'s own `LetPrim::ty` is always `()` (its bound result is unit and
/// never read, see `lower_prim_op`'s own doc comment).
fn lower_array_store<'c>(
    ctx: &LowerCtx<'c, '_>,
    block: &Block<'c>,
    env: &HashMap<CVar, Value<'c, 'c>>,
    array_ty: &Ty,
    args: &[CVal],
) {
    let CVal::Var(array_var) = &args[0] else {
        panic!("MLIR lowering: `store`'s own array operand must be a variable");
    };
    let array_val = *env
        .get(array_var)
        .unwrap_or_else(|| panic!("MLIR lowering: unbound CPS variable v{array_var}"));
    let Some((value_arg, index_args)) = args[1..].split_last() else {
        panic!("MLIR lowering: `store` needs at least an index and a value");
    };
    let i32_ty = width_ty(ctx, "i32");
    let location = Location::unknown(ctx.context);
    if array_val.r#type().is_tensor() {
        // See `lower_array_load`'s identical check: a native `tensor<...>`
        // SSA value is purely functional (`tensor.insert` would produce a
        // *new* value, not mutate this one in place) and nothing here
        // rebinds the CPS variable this array came from to that new value
        // — out of scope for now, panics clearly rather than silently
        // discarding the write.
        panic!(
            "MLIR lowering: index-assignment into a `#[mlir_type(...)]`-tagged native value isn't supported yet"
        );
    } else if array_val.r#type().is_mem_ref() {
        let elem_ty = MemRefType::try_from(array_val.r#type())
            .unwrap_or_else(|_| panic!("MLIR lowering: `store`'s own array operand isn't a memref"))
            .element();
        let index_vals: Vec<Value> = index_args
            .iter()
            .map(|a| to_index(ctx, block, lower_cval(ctx.context, block, env, a, i32_ty)))
            .collect();
        let value_val = lower_cval(ctx.context, block, env, value_arg, elem_ty);
        block.append_operation(memref::store(value_val, array_val, &index_vals, location));
    } else {
        let (_, leaf_ty) = flatten_array_dims(array_ty);
        let array_llvm_ty = ty_to_llvm_field_type(ctx, array_ty);
        let elem_mlir_ty = ty_to_mlir(ctx, leaf_ty);
        let mut gep_indices = vec![const_i32(ctx, block, 0)];
        gep_indices.extend(
            index_args
                .iter()
                .map(|a| lower_cval(ctx.context, block, env, a, i32_ty)),
        );
        let leaf_ptr = gep_dynamic(ctx, block, array_val, &gep_indices, array_llvm_ty);
        let value_val = lower_cval(ctx.context, block, env, value_arg, elem_mlir_ty);
        block.append_operation(llvm::store(
            ctx.context,
            value_val,
            leaf_ptr,
            location,
            LoadStoreOptions::new(),
        ));
    }
}

/// `PrimOp::Struct(name, field_names)` — `field_names` is whatever order the
/// *literal* itself wrote (named construction, `Complex(imag: .., real: ..)`,
/// can differ from declaration order), which is fine: each field is stored
/// at its own *declared* `position` (`struct_field_types`'s own canonical
/// order), not at its position within `field_names` — the GEP index, not
/// insertion order, is what actually determines the resulting layout.
/// Allocates one `llvm.alloca`'d slot up front (`alloc_llvm_value`) and returns
/// its own pointer — see `struct_llvm_type`'s own doc comment for why a
/// struct is reference-, not value-, typed here. An array-typed field is
/// filled by copying its already-built standalone value (either
/// representation — `memref` or, for a struct leaf, `!llvm.ptr` — see
/// `array_leaf_is_struct`) element-by-element into the struct's own
/// *embedded* `!llvm.array` slot (`copy_array_into_llvm_field`) — not stored
/// as a reference, since neither standalone representation is itself a
/// valid `!llvm.struct` field type (see `struct_llvm_type`'s own doc comment
/// on why a `memref` can't be one, and `ty_to_mlir`'s own `Ty::Array` arm
/// doc comment for the analogous reason a struct-leaf array's own top-level
/// `!llvm.ptr` still isn't the *field*'s own inline `!llvm.array` shape).
fn lower_struct_construct<'c>(
    ctx: &LowerCtx<'c, '_>,
    block: &Block<'c>,
    env: &HashMap<CVar, Value<'c, 'c>>,
    ty: &Ty,
    field_names: &[String],
    args: &[CVal],
) -> Value<'c, 'c> {
    let (name, type_args) = struct_name_and_args(ty);
    let field_types = struct_field_types(&ctx.struct_schemas, name, type_args);
    let struct_llvm_ty = struct_llvm_type(ctx, name, type_args);
    let ptr = alloc_llvm_value(ctx, block, struct_llvm_ty);
    for (field_name, arg) in field_names.iter().zip(args) {
        let position = field_types
            .iter()
            .position(|(n, _)| n == field_name)
            .unwrap_or_else(|| {
                panic!("MLIR lowering: struct `{name}` has no field `{field_name}`")
            });
        let (_, field_ty) = &field_types[position];
        let field_ptr = gep(ctx, block, ptr, &[0, position as i64], struct_llvm_ty);
        store_field(ctx, block, env, field_ty, field_ptr, arg);
    }
    ptr
}

/// `PrimOp::Struct` construction for a `#[mlir_type(tensor)]`/`#[mlir_type(
/// vector)]`-tagged struct (`native_shape_keyword`/`tagged_struct_native_
/// type`'s own doc comment has the full design) — a genuinely different
/// path from `lower_struct_construct`, not a variant of it: there's no
/// `!llvm.struct` allocation at all, the result is a bare native SSA value.
/// The struct's own sole field (already checked array-typed by `tagged_
/// struct_native_type`) is constructed exactly like any other standalone
/// array (`lower_array_construct`, unchanged, already ran to produce
/// `args`' own single already-lowered `memref` value) — this function's
/// only job is reading every one of its scalar elements back out
/// (`flatten_memref_elements`, row-major, the read-side mirror of `copy_
/// nested_array`'s own write-side walk) and collecting them into one
/// `{keyword}.from_elements`. Deliberately not optimized to skip the
/// memref round-trip even when the field's own value expression is
/// syntactically a literal — "let MLIR have fun with the optimization"
/// was an explicit, deliberate call: this stays one uniform path
/// regardless of where the array value came from (a literal, a computed
/// expression, a variable), rather than special-casing the literal shape
/// the way `Vector`'s own now-removed reserved-call mechanism used to.
fn lower_tagged_struct_construct<'c>(
    ctx: &LowerCtx<'c, '_>,
    block: &Block<'c>,
    env: &HashMap<CVar, Value<'c, 'c>>,
    ty: &Ty,
    args: &[CVal],
) -> Value<'c, 'c> {
    let (name, type_args) = struct_name_and_args(ty);
    let keyword = native_shape_keyword(ctx, name).unwrap_or_else(|| {
        panic!("MLIR lowering: `{name}` has no recognized #[mlir_type(...)] shape keyword")
    });
    let fields = struct_field_types(&ctx.struct_schemas, name, type_args);
    let [(_, field_ty)] = fields.as_slice() else {
        panic!(
            "MLIR lowering: `#[mlir_type({keyword})]` requires exactly one field, `{name}` has {}",
            fields.len()
        );
    };
    let [field_arg] = args else {
        panic!(
            "MLIR lowering: `{name}` construction expects exactly one argument (its own sole field), got {}",
            args.len()
        );
    };
    let src = lower_nested_array_arg(env, field_arg);
    let (dims, _leaf_ty) = flatten_array_dims(field_ty);
    let mut elems = Vec::new();
    flatten_memref_elements(ctx, block, src, &dims, &mut elems);

    // A region-local variant of this construction was tried here (build the
    // memref explicitly via `cleave_alloc_local`, skip `tensor.from_
    // elements` entirely — closing a real gap: `load_train_input`/`load_
    // train_target`, `region_analysis.rs`'s own tests confirm, get marked
    // region-local correctly but their own tensor-shaped return never
    // reached `cleave_alloc_local` through the *ordinary* path below,
    // `tensor.from_elements`'s own physical backing deferred all the way to
    // One-Shot Bufferize, long after `--inline` has erased which function it
    // came from). Reverted, not kept: measured directly (VTune, `examples/
    // mnist-interop`) to reintroduce ~63s of real `memcpy` traffic — the
    // hand-built memref (via `unrealized_conversion_cast` on a from-scratch
    // descriptor) is opaque to One-Shot Bufferize's own alias analysis in a
    // way `tensor.from_elements` itself isn't, so bufferization can no
    // longer prove downstream reads are copy-free the way it can for the
    // ordinary path — a real, measured regression, not a hypothetical one.
    // The struct-boundary half of the region-local mechanism (`alloc_llvm_
    // value`'s own doc comment) still stands; only this tensor-construction
    // extension specifically was undone.
    let native_ty = ty_to_mlir(ctx, ty);

    let location = Location::unknown(ctx.context);
    let built = OperationBuilder::new(&format!("{keyword}.from_elements"), location)
        .add_operands(&elems)
        .add_results(&[native_ty])
        .build()
        .unwrap_or_else(|e| {
            panic!("MLIR lowering: failed to build `{keyword}.from_elements`: {e}")
        });
    block.append_operation(built).result(0).unwrap().into()
}

/// Walks every leaf position of a (possibly multi-dim) memref `src`, in
/// row-major order, `memref.load`-ing each scalar element into `out` — the
/// read side of exactly the same walk `copy_nested_array` already does for
/// writing, every dimension fully unrolled (a compile-time constant,
/// cleave has no dynamically-sized arrays).
fn flatten_memref_elements<'c>(
    ctx: &LowerCtx<'c, '_>,
    block: &Block<'c>,
    src: Value<'c, 'c>,
    dims: &[i64],
    out: &mut Vec<Value<'c, 'c>>,
) {
    fn walk<'c>(
        ctx: &LowerCtx<'c, '_>,
        block: &Block<'c>,
        src: Value<'c, 'c>,
        remaining: &[i64],
        idx_acc: &mut Vec<Value<'c, 'c>>,
        out: &mut Vec<Value<'c, 'c>>,
    ) {
        let Some((&dim, rest)) = remaining.split_first() else {
            let location = Location::unknown(ctx.context);
            let load_op = block.append_operation(memref::load(src, idx_acc, location));
            out.push(load_op.result(0).unwrap().into());
            return;
        };
        for i in 0..dim {
            idx_acc.push(const_index(ctx, block, i));
            walk(ctx, block, src, rest, idx_acc, out);
            idx_acc.pop();
        }
    }
    walk(ctx, block, src, dims, &mut Vec::new(), out);
}

/// Stores `arg` into a field addressed by `field_ptr` (`field_ty` its own
/// concrete type) — shared by struct construction (`lower_struct_construct`,
/// one call per field) and direct field-mutation assignment
/// (`lower_field_store`, `s.field = v`). An array-typed field is copied
/// element-by-element from its already-built standalone source value (either
/// representation, see `copy_array_into_llvm_field`'s own doc comment) into
/// the struct's own *embedded* `!llvm.array` slot — see `struct_llvm_type`'s
/// own doc comment for why neither standalone representation can be an
/// `!llvm.struct` field directly, so it's never stored as a bare reference.
fn store_field<'c>(
    ctx: &LowerCtx<'c, '_>,
    block: &Block<'c>,
    env: &HashMap<CVar, Value<'c, 'c>>,
    field_ty: &Ty,
    field_ptr: Value<'c, 'c>,
    arg: &CVal,
) {
    if native_shape_field_keyword(ctx, field_ty).is_some() {
        store_native_shape_field(ctx, block, env, field_ty, field_ptr, arg);
    } else if is_array_ty(field_ty) {
        let src = lower_nested_array_arg(env, arg);
        let (dims, leaf_ty) = flatten_array_dims(field_ty);
        let field_llvm_ty = ty_to_llvm_field_type(ctx, field_ty);
        let elem_mlir_ty = ty_to_mlir(ctx, leaf_ty);
        copy_array_into_llvm_field(
            ctx,
            block,
            src,
            &dims,
            field_ptr,
            field_llvm_ty,
            elem_mlir_ty,
        );
    } else {
        let field_mlir_ty = ty_to_mlir(ctx, field_ty);
        let value = lower_cval(ctx.context, block, env, arg, field_mlir_ty);
        let location = Location::unknown(ctx.context);
        block.append_operation(llvm::store(
            ctx.context,
            value,
            field_ptr,
            location,
            LoadStoreOptions::new(),
        ));
    }
}

/// The exact LLVM struct layout MLIR's own `finalize-memref-to-llvm`
/// conversion (part of `pipeline.rs`'s own `to-llvm` pass stage) uses for a
/// statically-shaped memref's own descriptor — `(allocated_ptr, aligned_ptr,
/// offset, sizes[rank], strides[rank])` — confirmed directly, not guessed:
/// `mlir-opt --finalize-memref-to-llvm` on a real, standalone `memref.alloc`
/// probe (`I:/Dev/llvm-mlir-22/bin/mlir-opt.exe`, this exact toolchain) prints
/// this exact shape. Letting a `Tensor`/`Vector`-typed struct field *be* this
/// descriptor (`ty_to_llvm_field_type`'s own native-shape-tagged branch,
/// below) — rather than the field's own fully-flattened `!llvm.array` — is
/// what lets `store_native_shape_field`/`load_native_shape_field` become
/// real O(1) aggregate loads/stores instead of one op per element: a
/// `builtin.unrealized_conversion_cast` between a `memref<...>` value and
/// this exact struct shape is a genuine identity materialization once
/// `finalize-memref-to-llvm` runs — confirmed directly, the same way: both
/// directions (`memref -> this struct`, `this struct -> memref`) resolve to
/// clean, cast-free code under `--finalize-memref-to-llvm --reconcile-
/// unrealized-casts`, no leftover `unrealized_conversion_cast` in the
/// output, not a guess about `egg`-style "should fold" wishful thinking.
pub(crate) fn memref_descriptor_llvm_type<'c>(context: &'c Context, rank: usize) -> Type<'c> {
    let ptr = Type::parse(context, "!llvm.ptr")
        .unwrap_or_else(|| panic!("MLIR lowering: failed to parse `!llvm.ptr`"));
    let i64_ty: Type = IntegerType::new(context, 64).into();
    let dims_ty = llvm::r#type::array(i64_ty, rank as u32);
    llvm::r#type::r#struct(context, &[ptr, ptr, i64_ty, dims_ty, dims_ty], false)
}

/// Stores a `Tensor`/`Vector`-typed field's own value (a real native SSA
/// value, e.g. `tensor<64x32xf32>` — never an `!llvm.struct` allocation of
/// its own, `lower_tagged_struct_construct`'s own doc comment) into its
/// embedded field storage — a real O(1) aggregate store, not one `tensor.
/// extract`+`llvm.store` pair per element. Found necessary directly: the
/// original per-element version (kept working, numerically verified, for a
/// long stretch of this project) turned out to dominate real compile time at
/// real network scale (`doc/backlog.md`'s own "digits-interop" perf item) —
/// `Optimizer::step<Sgd,Network>` alone reached 34,033 lines of MLIR for one
/// real 64x32/32x10 two-layer network, almost entirely this same per-element
/// pattern, in *both* directions, at every struct-field crossing a `Tensor`
/// value makes (`Dense`'s own `w`/`b`, `Network`'s own `l1`/`l2`, layered on
/// top of each other for `Optimizer::step`'s own field-by-field recursion).
///
/// Real fix, not a workaround: `bufferization.to_buffer` (tensor -> memref,
/// one op, the exact reverse of `Ring<Tensor<T,Dims...>>::zero()`'s own
/// `tensor.splat`-adjacent `bufferization.to_tensor` fix — see `stdlib/nn/
/// nn.cleave`'s `Init<Dense<T,In,Out>>::xavier`/`he`, which already uses `to_
/// tensor` from cleave source directly), then a copy into a fresh, `cleave_
/// alloc_rc`'d buffer (see below for why), then one ordinary aggregate
/// `llvm.store` of a hand-built descriptor — no different in kind from
/// storing any other non-array field.
///
/// **Why the payload itself is copied into a `cleave_alloc_rc`'d buffer,
/// not just cast in place the way `load_native_shape_field` used to — found
/// by direct testing, a real, load-bearing fix, not caution for its own
/// sake.** Casting `bufferization.to_buffer`'s own result straight into the
/// field's descriptor bits (this function's own earlier form) makes the
/// *only* remaining real use of that memref, from `--buffer-deallocation-
/// pipeline`'s own point of view, the opaque `unrealized_conversion_cast` —
/// invisible to it, so it concludes the memref has no further use and
/// inserts a `memref.dealloc` for it *immediately after this store*,
/// confirmed directly (`mlir-opt`, this exact toolchain): the struct's own
/// field is left holding a pointer to memory that was freed the very next
/// instruction. A `memref.alloc`'d buffer can't be told "you're wrong, this
/// one's mine" after the fact — `load_native_shape_field`'s own read-side
/// fix works by making the *promise* to that pass true (a genuinely
/// exclusive copy); there's no equivalent move here, since the struct's own
/// field storage is what needs to keep the data *alive*, past the point
/// where any MLIR-tracked memref could still validly represent it. The
/// real fix: don't let `--buffer-deallocation-pipeline` ever see the buffer
/// that ends up in the struct's own field at all — allocate it through
/// `cleave_alloc_rc` (`alloc_llvm_value`, the exact same choke point every
/// struct's own allocation already goes through) instead of `memref.alloc`,
/// copy the computed value's own data into it via a raw `llvm.intr.memcpy`
/// (bypassing `memref.copy`, which needs a real `memref`-typed destination
/// — this one is deliberately *not* one), then hand-build the field's own
/// descriptor bits directly (`memref_descriptor_llvm_type`'s own confirmed
/// layout: `allocated_ptr`/`aligned_ptr` both the fresh pointer — `cleave_
/// alloc_rc` already hands back one ready-to-use pointer, no separate
/// alignment-rounding step the way raw `malloc` needs — `offset` zero,
/// `sizes`/`strides` both compile-time constants, row-major, since cleave
/// has no dynamically-shaped tensors to begin with). The *source* memref
/// (`bufferization.to_buffer`'s own result) stays entirely ordinary,
/// `memref.alloc`/`--buffer-deallocation-pipeline`-tracked exactly like any
/// other intermediate tensor value — correctly freed once the `memcpy`
/// reads its own last byte, never touching the struct's own field.
fn store_native_shape_field<'c>(
    ctx: &LowerCtx<'c, '_>,
    block: &Block<'c>,
    env: &HashMap<CVar, Value<'c, 'c>>,
    field_ty: &Ty,
    field_ptr: Value<'c, 'c>,
    arg: &CVal,
) {
    let (name, type_args) = struct_name_and_args(field_ty);
    let keyword = native_shape_keyword(ctx, name)
        .expect("caller already confirmed this is native-shape-tagged");
    let fields = struct_field_types(&ctx.struct_schemas, name, type_args);
    let [(_, inner_ty)] = fields.as_slice() else {
        panic!(
            "MLIR lowering: `#[mlir_type({keyword})]` requires exactly one field, `{name}` has {}",
            fields.len()
        );
    };
    // See `load_native_shape_field`'s own identical assertion for why.
    assert_eq!(
        keyword, "tensor",
        "MLIR lowering: O(1) native-shape field storage needs a real memref-backed form, which `#[mlir_type(vector)]` doesn't have"
    );
    let (dims, leaf_ty) = flatten_array_dims(inner_ty);
    let native_ty = ty_to_mlir(ctx, field_ty);
    let value = lower_cval(ctx.context, block, env, arg, native_ty);
    let elem_mlir_ty = ty_to_mlir(ctx, leaf_ty);
    let context = ctx.context;
    let location = Location::unknown(context);

    let memref_ty: Type = MemRefType::new(elem_mlir_ty, &dims, None, None).into();
    // `bufferization.to_buffer`, not the older `to_memref` name some MLIR
    // docs/versions use — confirmed directly against this exact toolchain
    // (`mlir-opt`, `I:/Dev/llvm-mlir-22`): `to_memref` verifies as an
    // unregistered op here, `to_buffer` is this version's real name for the
    // identical tensor -> memref direction (`bufferization.to_tensor`'s own
    // real, unrenamed counterpart).
    let to_buffer = OperationBuilder::new("bufferization.to_buffer", location)
        .add_operands(&[value])
        .add_results(&[memref_ty])
        .build()
        .unwrap_or_else(|e| panic!("MLIR lowering: failed to build bufferization.to_buffer: {e}"));
    let memref_val: Value = block.append_operation(to_buffer).result(0).unwrap().into();

    // Source data pointer — the identical extraction `array_ptr_and_len`
    // already uses for the unrelated "array crossing an extern fn boundary"
    // case, reused here as-is.
    let index_ty = Type::index(context);
    let extract = OperationBuilder::new("memref.extract_aligned_pointer_as_index", location)
        .add_operands(&[memref_val])
        .add_results(&[index_ty])
        .build()
        .unwrap_or_else(|e| {
            panic!("MLIR lowering: failed to build memref.extract_aligned_pointer_as_index: {e}")
        });
    let src_idx: Value = block.append_operation(extract).result(0).unwrap().into();
    let i64_ty: Type = IntegerType::new(context, 64).into();
    let src_i64: Value = block
        .append_operation(arith::index_cast(src_idx, i64_ty, location))
        .result(0)
        .unwrap()
        .into();
    let ptr_ty = llvm::r#type::pointer(context, 0);
    let src_ptr: Value = block
        .append_operation(
            OperationBuilder::new("llvm.inttoptr", location)
                .add_operands(&[src_i64])
                .add_results(&[ptr_ty])
                .build()
                .unwrap_or_else(|e| panic!("MLIR lowering: failed to build llvm.inttoptr: {e}")),
        )
        .result(0)
        .unwrap()
        .into();

    // Fresh, `cleave_alloc_rc`'d destination — sized as a flat `!llvm.array`
    // of every element, matching `alloc_llvm_value`'s own generic "any LLVM
    // type" contract exactly the way a struct-leaf array already uses it.
    let total_elems: u32 = dims.iter().product::<i64>() as u32;
    let flat_array_ty = llvm::r#type::array(elem_mlir_ty, total_elems);
    let dest_ptr = alloc_llvm_value(ctx, block, flat_array_ty);
    let size = llvm_type_size_bytes(ctx, block, flat_array_ty);
    let is_volatile = Attribute::parse(context, "false")
        .unwrap_or_else(|| panic!("MLIR lowering: failed to parse `false` attribute"));
    block.append_operation(
        OperationBuilder::new("llvm.intr.memcpy", location)
            .add_operands(&[dest_ptr, src_ptr, size])
            .add_attributes(&[(Identifier::new(context, "isVolatile"), is_volatile)])
            .build()
            .unwrap_or_else(|e| panic!("MLIR lowering: failed to build llvm.intr.memcpy: {e}")),
    );

    // Hand-built descriptor — `memref_descriptor_llvm_type`'s own confirmed
    // `(allocated_ptr, aligned_ptr, offset, sizes[rank], strides[rank])`
    // layout, row-major strides (cleave's own tensors are always
    // statically, fully shaped — no dynamic dimension to account for).
    let descriptor_ty = memref_descriptor_llvm_type(context, dims.len());
    let zero_i64: Value = block
        .append_operation(arith::constant(
            context,
            IntegerAttribute::new(i64_ty, 0).into(),
            location,
        ))
        .result(0)
        .unwrap()
        .into();
    let mut descriptor_val: Value = block
        .append_operation(llvm::poison(descriptor_ty, location))
        .result(0)
        .unwrap()
        .into();
    for pos in [0i64, 1] {
        descriptor_val = block
            .append_operation(llvm::insert_value(
                context,
                descriptor_val,
                DenseI64ArrayAttribute::new(context, &[pos]),
                dest_ptr,
                location,
            ))
            .result(0)
            .unwrap()
            .into();
    }
    descriptor_val = block
        .append_operation(llvm::insert_value(
            context,
            descriptor_val,
            DenseI64ArrayAttribute::new(context, &[2]),
            zero_i64,
            location,
        ))
        .result(0)
        .unwrap()
        .into();
    let mut stride = 1i64;
    let mut strides = vec![0i64; dims.len()];
    for i in (0..dims.len()).rev() {
        strides[i] = stride;
        stride *= dims[i];
    }
    for (i, &dim) in dims.iter().enumerate() {
        let dim_val: Value = block
            .append_operation(arith::constant(
                context,
                IntegerAttribute::new(i64_ty, dim).into(),
                location,
            ))
            .result(0)
            .unwrap()
            .into();
        descriptor_val = block
            .append_operation(llvm::insert_value(
                context,
                descriptor_val,
                DenseI64ArrayAttribute::new(context, &[3, i as i64]),
                dim_val,
                location,
            ))
            .result(0)
            .unwrap()
            .into();
        let stride_val: Value = block
            .append_operation(arith::constant(
                context,
                IntegerAttribute::new(i64_ty, strides[i]).into(),
                location,
            ))
            .result(0)
            .unwrap()
            .into();
        descriptor_val = block
            .append_operation(llvm::insert_value(
                context,
                descriptor_val,
                DenseI64ArrayAttribute::new(context, &[4, i as i64]),
                stride_val,
                location,
            ))
            .result(0)
            .unwrap()
            .into();
    }

    block.append_operation(llvm::store(
        context,
        descriptor_val,
        field_ptr,
        location,
        LoadStoreOptions::new(),
    ));
}

/// Reads a `Tensor`/`Vector`-typed field's own value back out of its
/// embedded field storage — the read-side mirror of `store_native_shape_
/// field`'s own doc comment: one ordinary aggregate `llvm.load`, `builtin.
/// unrealized_conversion_cast` back to `memref<...>`, then straight into
/// `bufferization.to_tensor ... restrict` (`restrict` alone, no `writable`
/// -- see below for why this, not a defensive copy, is the real fix) — no
/// per-element `llvm.getelementptr`+`llvm.load` walk at all, and (unlike an
/// earlier version of this function) no copy at all either.
///
/// **`restrict` alone, without `writable`, is both required and
/// sufficient — found by direct testing against this exact toolchain, not
/// assumed.** `restrict` is mandatory: One-Shot Analysis rejects a bare
/// `to_tensor` outright ("to_tensor ops without `restrict` are not
/// supported"), so there is no weaker option to fall back to. The real
/// question this function's own earlier version got wrong was pairing it
/// with `writable` unconditionally — `writable` is what actually invites
/// both failure modes a defensive copy used to exist to prevent: (1)
/// **silent data corruption** — with `writable`, One-Shot Bufferize is
/// free to compute some *other* op's result straight back into this same
/// buffer, in place (confirmed directly: a `linalg.generic` consuming this
/// value wrote its own result into the struct's own storage); (2) once
/// `--buffer-deallocation-pipeline` runs, a **real `STATUS_ACCESS_
/// VIOLATION`** — the pass, trusting the "exclusively *owned*" half of the
/// promise, frees the struct's own storage once this value's own last use
/// passes. Neither risk is real for a value that's genuinely never written
/// to: a field read, in cleave's own always-reconstruct-never-mutate
/// discipline (`doc/hld.md`'s own `struct_llvm_type` doc comment), is
/// *never* the target of an in-place write anywhere downstream — only
/// `writable` ever grants that permission in the first place, `restrict`
/// alone just says "nothing else aliases *this specific SSA value*",
/// which is true regardless of how many separate reads of the same
/// underlying field exist elsewhere, each with its own independent cast.
/// Confirmed directly, by hand, against this exact toolchain (`mlir-opt`):
/// a `restrict`-but-not-`writable` `to_tensor` is used as-is, with no
/// buffer materialized for it, by a consuming `linalg.generic`'s own
/// `ins()`, and `--buffer-deallocation-pipeline` never inserts a `memref.
/// dealloc` for it at all — only for the *other*, genuinely-owned buffers
/// (`tensor.empty()`-seeded intermediates) in the same function.
fn load_native_shape_field<'c>(
    ctx: &LowerCtx<'c, '_>,
    block: &Block<'c>,
    field_ty: &Ty,
    field_ptr: Value<'c, 'c>,
) -> Value<'c, 'c> {
    let (name, type_args) = struct_name_and_args(field_ty);
    let keyword = native_shape_keyword(ctx, name)
        .expect("caller already confirmed this is native-shape-tagged");
    let fields = struct_field_types(&ctx.struct_schemas, name, type_args);
    let [(_, inner_ty)] = fields.as_slice() else {
        panic!(
            "MLIR lowering: `#[mlir_type({keyword})]` requires exactly one field, `{name}` has {}",
            fields.len()
        );
    };
    let (dims, leaf_ty) = flatten_array_dims(inner_ty);
    let elem_mlir_ty = ty_to_mlir(ctx, leaf_ty);
    let location = Location::unknown(ctx.context);

    let descriptor_ty = memref_descriptor_llvm_type(ctx.context, dims.len());
    let descriptor_val: Value = block
        .append_operation(llvm::load(
            ctx.context,
            field_ptr,
            descriptor_ty,
            location,
            LoadStoreOptions::new(),
        ))
        .result(0)
        .unwrap()
        .into();

    let memref_ty: Type = MemRefType::new(elem_mlir_ty, &dims, None, None).into();
    let cast = OperationBuilder::new("builtin.unrealized_conversion_cast", location)
        .add_operands(&[descriptor_val])
        .add_results(&[memref_ty])
        .build()
        .unwrap_or_else(|e| {
            panic!("MLIR lowering: failed to build unrealized_conversion_cast: {e}")
        });
    let memref_val: Value = block.append_operation(cast).result(0).unwrap().into();

    let native_ty = ty_to_mlir(ctx, field_ty);
    let restrict = Attribute::parse(ctx.context, "unit")
        .unwrap_or_else(|| panic!("MLIR lowering: failed to parse `unit` attribute"));
    // `bufferization.to_tensor`/`to_buffer` (`store_native_shape_field`'s own
    // doc comment) are specific to the `tensor`/`memref` pair, not a
    // `{keyword}`-generic pair the way `{keyword}.from_elements` above used
    // to be — `#[mlir_type(vector)]` (structurally supported, currently
    // unused anywhere in stdlib) has no memref-backed form at all, so this
    // whole O(1) path is real only for `keyword == "tensor"`.
    assert_eq!(
        keyword, "tensor",
        "MLIR lowering: O(1) native-shape field access needs a real memref-backed form, which `#[mlir_type(vector)]` doesn't have"
    );
    // `restrict`, deliberately alone (no `writable`) -- see this function's
    // own doc comment for the full story.
    let to_tensor = OperationBuilder::new("bufferization.to_tensor", location)
        .add_operands(&[memref_val])
        .add_attributes(&[(Identifier::new(ctx.context, "restrict"), restrict)])
        .add_results(&[native_ty])
        .build()
        .unwrap_or_else(|e| panic!("MLIR lowering: failed to build bufferization.to_tensor: {e}"));
    block.append_operation(to_tensor).result(0).unwrap().into()
}

/// `PrimOp::FieldStore { struct_ty, field }`, `args = [base, value]` — a
/// direct field-mutation assignment (`s.field = v`), mirroring `PrimOp::
/// Store`'s own "real effect, bound result unit and never read" shape for
/// arrays (see `lower_array_store`'s own doc comment): the struct's own
/// identity never changes, only the one field's own storage, addressed via
/// the exact same GEP `lower_struct_construct`/`lower_field_access` already
/// use.
fn lower_field_store<'c>(
    ctx: &LowerCtx<'c, '_>,
    block: &Block<'c>,
    env: &HashMap<CVar, Value<'c, 'c>>,
    struct_ty: &Ty,
    field: &str,
    args: &[CVal],
) {
    let CVal::Var(base_var) = &args[0] else {
        panic!("MLIR lowering: field mutation's own base operand must be a variable");
    };
    let base_val = *env
        .get(base_var)
        .unwrap_or_else(|| panic!("MLIR lowering: unbound CPS variable v{base_var}"));
    let (name, type_args) = struct_name_and_args(struct_ty);
    // See `lower_field_access`'s identical check: a tag-tensor struct has no
    // address of its own to store into at all.
    if native_shape_keyword(ctx, name).is_some() {
        panic!(
            "MLIR lowering: `{name}` is a `#[mlir_type(...)]`-tagged native value, its own field can't be mutated in place"
        );
    }
    let field_types = struct_field_types(&ctx.struct_schemas, name, type_args);
    let position = field_types
        .iter()
        .position(|(n, _)| n == field)
        .unwrap_or_else(|| panic!("MLIR lowering: struct `{name}` has no field `{field}`"));
    let (_, field_ty) = &field_types[position];
    let struct_llvm_ty = struct_llvm_type(ctx, name, type_args);
    let field_ptr = gep(ctx, block, base_val, &[0, position as i64], struct_llvm_ty);
    store_field(ctx, block, env, field_ty, field_ptr, &args[1]);
}

/// `PrimOp::Field { struct_ty, field }`, `args = [base]` — `struct_ty` is
/// the base expression's own concrete cleave type (see `PrimOp::Field`'s own
/// doc comment for why this can't be recovered from the already-lowered
/// `Value` alone: an opaque `!llvm.ptr` carries no cleave-level struct name
/// or field layout of its own). A scalar/struct-typed field is *loaded* —
/// the read gives back the field's own current value (a nested struct's own
/// value is itself a pointer, per `struct_llvm_type`'s own doc comment, so
/// "loading" it just reads that pointer out). An **array-typed** field is
/// *not* loaded at all — `llvm.getelementptr`'s own result (the address of
/// the field's own embedded `!llvm.array`, still inside this struct's
/// storage) is returned directly, since a runtime index (`a.values[i,j]`,
/// `PrimOp::Load`/`Store`) needs an address to GEP further into, not a
/// (potentially huge) copied-out value.
fn lower_field_access<'c>(
    ctx: &LowerCtx<'c, '_>,
    block: &Block<'c>,
    env: &HashMap<CVar, Value<'c, 'c>>,
    struct_ty: &Ty,
    field: &str,
    args: &[CVal],
) -> Value<'c, 'c> {
    let CVal::Var(base_var) = &args[0] else {
        panic!("MLIR lowering: field access's own base operand must be a variable");
    };
    let base_val = *env
        .get(base_var)
        .unwrap_or_else(|| panic!("MLIR lowering: unbound CPS variable v{base_var}"));
    let (name, type_args) = struct_name_and_args(struct_ty);
    // A `#[mlir_type(tensor)]`/`#[mlir_type(vector)]`-tagged struct has no
    // `!llvm.struct` storage at all (`lower_tagged_struct_construct`'s own
    // doc comment) — its sole field (enforced array-typed, exactly one, by
    // `tagged_struct_native_type`) *is* the struct's own native SSA value,
    // so accessing it is a bare identity, never a real GEP into anything.
    if native_shape_keyword(ctx, name).is_some() {
        return base_val;
    }
    let field_types = struct_field_types(&ctx.struct_schemas, name, type_args);
    let position = field_types
        .iter()
        .position(|(n, _)| n == field)
        .unwrap_or_else(|| panic!("MLIR lowering: struct `{name}` has no field `{field}`"));
    let (_, field_ty) = &field_types[position];
    let struct_llvm_ty = struct_llvm_type(ctx, name, type_args);
    let field_ptr = gep(ctx, block, base_val, &[0, position as i64], struct_llvm_ty);
    if native_shape_field_keyword(ctx, field_ty).is_some() {
        load_native_shape_field(ctx, block, field_ty, field_ptr)
    } else if is_array_ty(field_ty) {
        field_ptr
    } else {
        let result_ty = ty_to_mlir(ctx, field_ty);
        let location = Location::unknown(ctx.context);
        block
            .append_operation(llvm::load(
                ctx.context,
                field_ptr,
                result_ty,
                location,
                LoadStoreOptions::new(),
            ))
            .result(0)
            .unwrap()
            .into()
    }
}

/// `PrimOp::Retain(rc_ty)`/`PrimOp::Release(rc_ty)`, `args = [ptr]` — a real
/// call to `cleave_retain`/`cleave_release` (`cleave-rt`), the runtime half
/// of `refcount::insert_refcounting`'s own analysis (that module's own doc
/// comment has the full design). `rc_ty` only matters here for `ensure_
/// extern_declared`'s own signature -- any struct `Ty` maps to `!llvm.ptr`
/// via `ty_to_mlir`'s generic struct fallback, so the *specific* struct
/// name is irrelevant to the declared C signature, just needed because
/// `ensure_extern_declared` takes cleave-level `Ty`s, not raw MLIR types,
/// like every other extern declaration in this file. `args[0]` is always a
/// `CVal::Var` in practice (a struct-typed value is never a bare literal in
/// this IR), read directly out of `env` — no need to re-lower it through
/// `lower_cval`, its own `!llvm.ptr` value is already sitting there from
/// wherever it was constructed.
fn lower_refcount_call<'c>(
    ctx: &LowerCtx<'c, '_>,
    block: &Block<'c>,
    env: &HashMap<CVar, Value<'c, 'c>>,
    symbol: &str,
    rc_ty: &Ty,
    args: &[CVal],
) {
    let CVal::Var(ptr_var) = &args[0] else {
        panic!("MLIR lowering: `{symbol}`'s own operand must be a variable");
    };
    let ptr_val = *env
        .get(ptr_var)
        .unwrap_or_else(|| panic!("MLIR lowering: unbound CPS variable v{ptr_var}"));
    let context = ctx.context;
    let location = Location::unknown(context);
    ensure_extern_declared(ctx, symbol, std::slice::from_ref(rc_ty), &[]);
    block.append_operation(func::call(
        context,
        FlatSymbolRefAttribute::new(context, symbol),
        &[ptr_val],
        &[],
        location,
    ));
}

/// A raw, already-lowered `!llvm.ptr` version of `lower_refcount_call`'s
/// own `cleave_release` half — no `CVal`/`env` lookup, for the intermediate
/// pointers `lower_release_cascade` reads directly out of a struct's own
/// fields (never bound to a real CPS variable of their own). `rc_ty` only
/// matters for `ensure_extern_declared`'s own signature — see `lower_
/// refcount_call`'s own doc comment for why any struct `Ty` works equally
/// well there. Returns the real `i1` result `cleave_release` (`cleave-rt`)
/// now hands back — whether *this* call actually freed the allocation
/// (refcount reached zero) — `lower_release_cascade`'s own doc comment has
/// the full story for why this is load-bearing, not just informational.
fn emit_cleave_release<'c>(
    ctx: &LowerCtx<'c, '_>,
    block: &Block<'c>,
    rc_ty: &Ty,
    ptr_val: Value<'c, 'c>,
) -> Value<'c, 'c> {
    let context = ctx.context;
    let location = Location::unknown(context);
    let bool_ty: Type = IntegerType::new(context, 1).into();
    ensure_extern_declared(ctx, "cleave_release", std::slice::from_ref(rc_ty), &[bool_ty]);
    let call_op = block.append_operation(func::call(
        context,
        FlatSymbolRefAttribute::new(context, "cleave_release"),
        &[ptr_val],
        &[bool_ty],
        location,
    ));
    call_op.result(0).unwrap().into()
}

/// Cascades a struct's own release into every refcounted field it holds,
/// then releases the struct itself — the general fix `PrimOp::Release`'s
/// own doc comment (`cps.rs`) flags as still needed: `cleave_release` on
/// its own is flat, and a struct's own tensor-typed field, since `store_
/// native_shape_field`'s own fix, is *always* an independently `cleave_
/// alloc_rc`'d payload uniquely owned by that one field (tensors are
/// copied, never aliased, on every store — no retain-on-store needed for
/// them the way an *existing* struct value being embedded needs, `rewrite_
/// body`'s own doc comment) — releasing the container without also
/// releasing it just leaks it, unconditionally, on every single
/// replacement (found by direct testing: `examples/mnist-interop`'s own
/// real training run, `Optimizer::step` replacing `net`/`state` every
/// iteration, ran out of memory — `cleave_alloc_rc: allocation failed` —
/// once the tensor-payload leak this closes was the only one left).
///
/// Every child field's own pointer is read out *before* releasing
/// anything — the struct's own storage (and hence every field read through
/// it) becomes invalid the moment its own `cleave_release` call actually
/// frees it — but the cascade into those children only actually *runs*
/// inside a real `scf.if`, gated on `cleave_release`'s own returned `i1`
/// (`emit_cleave_release`'s own doc comment): whether the struct itself
/// was genuinely destroyed by *this* call, not merely had its own count
/// decremented while a second, still-live reference (another struct
/// embedding the exact same pointer, retained at construction time —
/// `rewrite_body`'s own retain-on-construction logic) keeps it alive.
/// **Load-bearing, found by direct testing, a real `STATUS_HEAP_
/// CORRUPTION`: cascading unconditionally — this function's own first
/// version — frees a nested struct's own tensor field the moment *any*
/// one of possibly several live references to it is released, not just
/// the last one** (`examples/digits-interop`'s own real `Optimizer::step`:
/// a freshly built `Dense` is retained again immediately, embedded into a
/// `Network`, then its own now-redundant local binding released — count 2
/// down to 1, very much still alive — the original, unconditional cascade
/// freed its tensor fields right there anyway). Children released depth-
/// first, the struct itself first-and-outermost this time (its own count
/// has to be known *before* deciding whether to touch its fields at all,
/// the reverse of naive destructor order) — safe regardless, since nothing
/// after this point ever reads through `ptr` again either way. A `#[mlir_
/// type(tensor)]`-tagged struct itself never reaches here directly
/// (`refcount::is_refcounted` excludes it — it has no `cleave_alloc_rc`'d
/// storage of its own to release, `lower_tagged_struct_construct`'s own
/// doc comment) — only as a *field* of another struct, the tensor-field
/// branch below.
fn lower_release_cascade<'c>(
    ctx: &LowerCtx<'c, '_>,
    block: &Block<'c>,
    struct_ty: &Ty,
    ptr: Value<'c, 'c>,
) {
    let (name, type_args) = struct_name_and_args(struct_ty);
    let field_types = struct_field_types(&ctx.struct_schemas, name, type_args);
    let struct_llvm_ty = struct_llvm_type(ctx, name, type_args);
    let context = ctx.context;
    let location = Location::unknown(context);

    enum PendingChild<'c> {
        Tensor(Value<'c, 'c>),
        Struct(Ty, Value<'c, 'c>),
    }
    let mut pending: Vec<PendingChild<'c>> = Vec::new();

    for (position, (_, field_ty)) in field_types.iter().enumerate() {
        let field_ptr = gep(ctx, block, ptr, &[0, position as i64], struct_llvm_ty);
        if let Some(keyword) = native_shape_field_keyword(ctx, field_ty) {
            assert_eq!(
                keyword, "tensor",
                "MLIR lowering: cascading release needs a real memref-backed form, which `#[mlir_type(vector)]` doesn't have"
            );
            // Read the field's own descriptor and pull its `allocated_ptr`
            // straight out (position 0, `memref_descriptor_llvm_type`'s own
            // confirmed layout) — no need for the full `load_native_shape_
            // field` machinery (`to_tensor`, the defensive copy) here,
            // this pointer is only ever handed to `cleave_release`, never
            // read as tensor data.
            let (fname, ftype_args) = struct_name_and_args(field_ty);
            let inner_fields = struct_field_types(&ctx.struct_schemas, fname, ftype_args);
            let [(_, inner_ty)] = inner_fields.as_slice() else {
                panic!(
                    "MLIR lowering: `#[mlir_type(tensor)]` requires exactly one field, `{fname}` has {}",
                    inner_fields.len()
                );
            };
            let (dims, _leaf_ty) = flatten_array_dims(inner_ty);
            let descriptor_ty = memref_descriptor_llvm_type(context, dims.len());
            let descriptor_val: Value = block
                .append_operation(llvm::load(
                    context,
                    field_ptr,
                    descriptor_ty,
                    location,
                    LoadStoreOptions::new(),
                ))
                .result(0)
                .unwrap()
                .into();
            let ptr_ty = llvm::r#type::pointer(context, 0);
            let base_ptr: Value = block
                .append_operation(llvm::extract_value(
                    context,
                    descriptor_val,
                    DenseI64ArrayAttribute::new(context, &[0]),
                    ptr_ty,
                    location,
                ))
                .result(0)
                .unwrap()
                .into();
            pending.push(PendingChild::Tensor(base_ptr));
        } else if matches!(field_ty, Ty::Con(_) | Ty::App(..))
            && ctx.struct_schemas.contains_key(struct_name_and_args(field_ty).0)
        {
            // An ordinary nested struct field — an opaque `!llvm.ptr`,
            // exactly like any other struct-typed value (`struct_llvm_
            // type`'s own doc comment) — read it, recurse once the
            // container's own fate is known.
            let child_ty = ty_to_mlir(ctx, field_ty);
            let child_val: Value = block
                .append_operation(llvm::load(
                    context,
                    field_ptr,
                    child_ty,
                    location,
                    LoadStoreOptions::new(),
                ))
                .result(0)
                .unwrap()
                .into();
            pending.push(PendingChild::Struct(field_ty.clone(), child_val));
        }
        // Else: a primitive/array-of-primitive field — nothing refcounted
        // to release.
    }

    let freed = emit_cleave_release(ctx, block, struct_ty, ptr);
    if pending.is_empty() {
        // No refcounted fields at all — the ordinary flat release above is
        // the whole story, no `scf.if` needed to gate an empty cascade.
        return;
    }

    let then_block = Block::new(&[]);
    for child in pending {
        match child {
            PendingChild::Tensor(child_ptr) => {
                // `struct_ty` (the *containing* struct), not the tensor
                // field's own type — this is only for `ensure_extern_
                // declared`'s own signature, and a `#[mlir_type(tensor)]`-
                // tagged type maps to a real `tensor<...>` under `ty_to_
                // mlir` (its own *native* MLIR type), wrong for `cleave_
                // release`'s own always-`!llvm.ptr` real C signature
                // (found by direct testing: a real verification failure,
                // `operand type mismatch: expected tensor<...>`).
                // `struct_ty` is guaranteed non-tensor-tagged here (this
                // function's own doc comment), so it maps to `!llvm.ptr`
                // correctly, exactly like every other `cleave_release`
                // call — tensors have no further nesting, the returned
                // `i1` is simply unused.
                emit_cleave_release(ctx, &then_block, struct_ty, child_ptr);
            }
            PendingChild::Struct(child_ty, child_val) => {
                lower_release_cascade(ctx, &then_block, &child_ty, child_val);
            }
        }
    }
    then_block.append_operation(scf::r#yield(&[], location));
    let then_region = Region::new();
    then_region.append_block(then_block);

    let else_block = Block::new(&[]);
    else_block.append_operation(scf::r#yield(&[], location));
    let else_region = Region::new();
    else_region.append_block(else_block);

    block.append_operation(scf::r#if(freed, &[], then_region, else_region, location));
}

/// Allocates one **heap**-backed slot shaped `llvm_ty`, returning its own
/// opaque `!llvm.ptr`, via a real call to `cleave_alloc_rc` (`cleave-rt`,
/// registered with the JIT the same way `print_i32`/... already are) — not
/// `cleave_alloc` any more: `doc/hld.md`'s own "Memory management" section,
/// Phase 0 (`cleave_alloc_rc`'s own doc comment in `cleave-rt` has the
/// fuller story). Every value this allocates now starts life with a real
/// refcount header (invisible to every existing GEP-based field access —
/// `cleave_alloc_rc` returns a pointer already offset *past* its own
/// header, byte-for-byte identical to what `cleave_alloc` used to hand
/// back), but nothing here inserts a matching `cleave_release` yet — that's
/// real, separate, higher-stakes work (deciding *where* a value's own scope
/// genuinely ends), not yet built. This swap alone changes no observable
/// behavior (still leaks, refcount frozen at whatever it started at,
/// verified directly against the full test suite) — it validates the
/// allocator swap is transparent before any release call is added on top.
/// Not `llvm.alloca`: found by direct testing — a struct returned from one
/// function and read by its own caller came back reading garbage, since an
/// `alloca`'d slot lives in *that function's own* stack frame, gone the
/// moment it returns, and a struct is a reference passed/returned by
/// pointer, never copied (see `struct_llvm_type`'s own doc comment) — its
/// storage has to outlive the call that built it. Fully generic over *any*
/// LLVM type, not struct-specific — also used to heap-allocate a struct-leaf
/// array's own embedded `!llvm.array` (`ty_to_mlir`'s `Ty::Array` arm,
/// `array_leaf_is_struct`), which needs the exact same "outlives the call
/// that built it" property a struct does.
/// **Picks `cleave_alloc_rc` vs `cleave_alloc_local` here, once, per
/// construction site** — the *only* place this decision gets made, since
/// `alloc_llvm_value` is already the one shared allocation primitive every
/// struct/array construction goes through (this function's own doc comment
/// above). `ctx.currently_region_local` (set once per top-level function,
/// `lower_top_level_fn`) is the deciding fact: `true` exactly when *this*
/// function's own name is in `region_analysis::find_region_local_
/// functions`'s own returned set — i.e., this function has exactly one
/// call site in the whole program, inside a loop, and its own result never
/// reaches that loop's own carried (escaping) state (`region_analysis.rs`'s
/// own module doc comment has the full reasoning). `cleave_alloc_local`'s
/// own `handle` parameter is passed as a literal `0` — never actually read
/// (`cleave-rt::cleave_alloc_local`'s own doc comment: correctness comes
/// from `REGION_DEPTH` being genuinely nonzero at the call, checked by
/// `assert_region_open`, not from the handle's own value) — so there's no
/// need to thread the real region handle `lower_loop`'s own `cleave_region_
/// enter` call returns all the way down to here.
fn alloc_llvm_value<'c>(
    ctx: &LowerCtx<'c, '_>,
    block: &Block<'c>,
    llvm_ty: Type<'c>,
) -> Value<'c, 'c> {
    let context = ctx.context;
    let location = Location::unknown(context);
    let ptr_ty = llvm::r#type::pointer(context, 0);
    let size = llvm_type_size_bytes(ctx, block, llvm_ty);
    let i64_ty: Type = IntegerType::new(context, 64).into();
    let (symbol, call_args): (&str, Vec<Value>) = if ctx.currently_region_local.get() {
        let zero_handle = block
            .append_operation(arith::constant(
                context,
                IntegerAttribute::new(i64_ty, 0).into(),
                location,
            ))
            .result(0)
            .unwrap()
            .into();
        ("cleave_alloc_local", vec![zero_handle, size])
    } else {
        ("cleave_alloc_rc", vec![size])
    };
    ensure_extern_declared(
        ctx,
        symbol,
        &vec![Ty::Con("i64".to_string()); call_args.len()],
        &[ptr_ty],
    );
    let call_op = block.append_operation(func::call(
        context,
        FlatSymbolRefAttribute::new(context, symbol),
        &call_args,
        &[ptr_ty],
        location,
    ));
    call_op.result(0).unwrap().into()
}

/// `llvm_ty`'s own byte size, as a real `i64` SSA value — the standard LLVM
/// IR idiom for `sizeof(T)` with no dedicated op needed: `getelementptr T,
/// null, 1` (one whole `T` past a null pointer) then `ptrtoint` gives
/// exactly the byte offset one `T` occupies, padding included, matching
/// LLVM's own real layout (hand-computing field/element offsets/alignment
/// here would risk silently disagreeing with it). Fully generic over any
/// LLVM type — `T` can be a struct or an `!llvm.array`, both go through
/// `alloc_llvm_value` the same way.
fn llvm_type_size_bytes<'c>(
    ctx: &LowerCtx<'c, '_>,
    block: &Block<'c>,
    llvm_ty: Type<'c>,
) -> Value<'c, 'c> {
    let context = ctx.context;
    let location = Location::unknown(context);
    let ptr_ty = llvm::r#type::pointer(context, 0);
    let null: Value = block
        .append_operation(llvm::zero(ptr_ty, location))
        .result(0)
        .unwrap()
        .into();
    let one_past = gep(ctx, block, null, &[1], llvm_ty);
    let i64_ty = IntegerType::new(context, 64).into();
    let built = OperationBuilder::new("llvm.ptrtoint", location)
        .add_operands(&[one_past])
        .add_results(&[i64_ty])
        .build()
        .unwrap_or_else(|e| panic!("MLIR lowering: failed to build llvm.ptrtoint: {e}"));
    block.append_operation(built).result(0).unwrap().into()
}

/// Extracts a bare `!llvm.ptr` from an already-lowered array value's own
/// data pointer, plus the array's own total element count as a real `i64`
/// — the pair an `extern fn`'s own array-typed parameter actually crosses
/// the call boundary as. Two possible *input* representations for the same
/// cleave-level array type, mirroring `copy_array_into_llvm_field`'s own
/// identical `is_mem_ref` branch (see its own doc comment): a standalone
/// array (a local/parameter) is a real `memref`, but the *same* array type
/// read back out of a struct/tuple field (`lower_field_access`'s own
/// `is_array_ty` branch) is already a bare `!llvm.ptr` into that struct's
/// own inline storage, never a `memref` at all — found by direct testing,
/// passing a tuple's own string-typed field (`x.0: [i8;N]`) straight into
/// `Print<[i8;N]>::print` (`examples`-motivated: `print(("x=", x))`).
/// Passing an already-bare pointer through `memref.extract_aligned_pointer_
/// as_index` fails MLIR verification outright (wrong operand kind) — no
/// extraction needed for it at all, it already *is* the value this
/// function exists to produce.
///
/// The `memref` branch: MLIR's default `convert-to-llvm` conversion (the
/// only pass this pipeline runs — melior's own binding exposes no options
/// on it, no bare-pointer-calling-convention toggle available) turns a
/// `memref` crossing any `func.call`/`llvm.call` boundary into a
/// descriptor *struct* (`{allocatedPtr, alignedPtr, offset, sizes[],
/// strides[]}`), passed by value — not a bare pointer. No hand-written
/// `cleave-rt` extern fn could plausibly match that layout, found by direct
/// testing (a hard `STATUS_ACCESS_VIOLATION` crash, not a clean panic,
/// before this fix). `memref.extract_aligned_pointer_as_index`/`llvm.
/// inttoptr` have no melior binding — built directly via `OperationBuilder`,
/// the identical pattern `llvm_type_size_bytes`'s own `llvm.ptrtoint`
/// already uses just above.
///
/// `len` is the array's own *total* element count (every dimension's own
/// size multiplied together, via `flatten_array_dims` — already exists for
/// exactly this "collapse a nested `Ty::Array` chain" need), a compile-time
/// constant materialized directly either way, never read off a runtime
/// descriptor.
fn array_ptr_and_len<'c>(
    ctx: &LowerCtx<'c, '_>,
    block: &Block<'c>,
    array_value: Value<'c, 'c>,
    array_ty: &Ty,
) -> (Value<'c, 'c>, Value<'c, 'c>) {
    let context = ctx.context;
    let location = Location::unknown(context);
    let i64_ty: Type = IntegerType::new(context, 64).into();

    let ptr: Value = if array_value.r#type().is_mem_ref() {
        let index_ty = Type::index(context);
        let built = OperationBuilder::new("memref.extract_aligned_pointer_as_index", location)
            .add_operands(&[array_value])
            .add_results(&[index_ty])
            .build()
            .unwrap_or_else(|e| {
                panic!(
                    "MLIR lowering: failed to build memref.extract_aligned_pointer_as_index: {e}"
                )
            });
        let idx: Value = block.append_operation(built).result(0).unwrap().into();
        let as_i64: Value = block
            .append_operation(arith::index_cast(idx, i64_ty, location))
            .result(0)
            .unwrap()
            .into();
        let ptr_ty = llvm::r#type::pointer(context, 0);
        let built = OperationBuilder::new("llvm.inttoptr", location)
            .add_operands(&[as_i64])
            .add_results(&[ptr_ty])
            .build()
            .unwrap_or_else(|e| panic!("MLIR lowering: failed to build llvm.inttoptr: {e}"));
        block.append_operation(built).result(0).unwrap().into()
    } else {
        array_value
    };

    let (dims, _leaf_ty) = flatten_array_dims(array_ty);
    let total: i64 = dims.iter().product();
    let len_attr = IntegerAttribute::new(i64_ty, total).into();
    let len: Value = block
        .append_operation(arith::constant(context, len_attr, location))
        .result(0)
        .unwrap()
        .into();

    (ptr, len)
}

/// Computes `llvm.getelementptr base[indices...]` for indices that are
/// *every one* a compile-time constant (a field GEP's own position, or a
/// fully-unrolled in-struct array-copy index, `copy_memref_into_llvm_field`)
/// — encoded directly into `rawConstantIndices` itself, **not** materialized
/// as `arith.constant` operands (found by direct testing: MLIR's own GEP
/// verifier rejects a struct field index that's merely a constant-*valued*
/// dynamic operand — "expected index N indexing a struct to be constant" —
/// it must be a real constant baked into the op itself; only `gep_dynamic`,
/// below, produces dynamic operands, used exclusively for genuinely runtime
/// array indices, which carry no such restriction).
fn gep<'c>(
    ctx: &LowerCtx<'c, '_>,
    block: &Block<'c>,
    base: Value<'c, 'c>,
    indices: &[i64],
    pointee_ty: Type<'c>,
) -> Value<'c, 'c> {
    let context = ctx.context;
    let location = Location::unknown(context);
    let ptr_ty = llvm::r#type::pointer(context, 0);
    let raw: Vec<i32> = indices.iter().map(|&i| i as i32).collect();
    let raw_indices = DenseI32ArrayAttribute::new(context, &raw);
    let built = OperationBuilder::new("llvm.getelementptr", location)
        .add_attributes(&[
            (
                Identifier::new(context, "rawConstantIndices"),
                raw_indices.into(),
            ),
            (
                Identifier::new(context, "elem_type"),
                TypeAttribute::new(pointee_ty).into(),
            ),
        ])
        .add_operands(&[base])
        .add_results(&[ptr_ty])
        .build()
        .unwrap_or_else(|e| panic!("MLIR lowering: failed to build llvm.getelementptr: {e}"));
    block.append_operation(built).result(0).unwrap().into()
}

/// Like `gep`, but for indices that are already runtime `Value`s (a loop
/// variable used as an array index) rather than compile-time constants —
/// built directly via `OperationBuilder` rather than melior's own `llvm::
/// get_element_ptr_dynamic` (which requires a compile-time-known index
/// *count*, via a const generic — this file's own callers need a genuinely
/// runtime-length index list, one per array dimension). `rawConstantIndices`
/// filled with `i32::MIN` sentinels throughout — melior's own convention
/// (confirmed in its `get_element_ptr_dynamic`) for "every index here is a
/// real operand, not a constant" — mixing compile-time-constant and dynamic
/// indices in one GEP is possible in principle but not needed by any caller
/// here, so not attempted.
fn gep_dynamic<'c>(
    ctx: &LowerCtx<'c, '_>,
    block: &Block<'c>,
    base: Value<'c, 'c>,
    indices: &[Value<'c, 'c>],
    pointee_ty: Type<'c>,
) -> Value<'c, 'c> {
    let context = ctx.context;
    let location = Location::unknown(context);
    let ptr_ty = llvm::r#type::pointer(context, 0);
    let raw_indices = DenseI32ArrayAttribute::new(context, &vec![i32::MIN; indices.len()]);
    let built = OperationBuilder::new("llvm.getelementptr", location)
        .add_attributes(&[
            (
                Identifier::new(context, "rawConstantIndices"),
                raw_indices.into(),
            ),
            (
                Identifier::new(context, "elem_type"),
                TypeAttribute::new(pointee_ty).into(),
            ),
        ])
        .add_operands(&[base])
        .add_operands(indices)
        .add_results(&[ptr_ty])
        .build()
        .unwrap_or_else(|e| panic!("MLIR lowering: failed to build llvm.getelementptr: {e}"));
    block.append_operation(built).result(0).unwrap().into()
}

fn const_i32<'c>(ctx: &LowerCtx<'c, '_>, block: &Block<'c>, n: i64) -> Value<'c, 'c> {
    let location = Location::unknown(ctx.context);
    let attribute = IntegerAttribute::new(width_ty(ctx, "i32"), n).into();
    let op = block.append_operation(arith::constant(ctx.context, attribute, location));
    op.result(0).unwrap().into()
}

/// Copies every scalar element of `src` — a flat, standalone array of shape
/// `dims`, either representation (`array_leaf_is_struct`): a `memref` (the
/// ordinary primitive/nested-array-leaf case) or an `!llvm.ptr` to a heap-
/// allocated `!llvm.array` (a struct leaf, `lower_array_construct`'s own
/// struct-leaf branch) — into the struct-embedded `!llvm.array` field
/// addressed by `field_ptr`, one load+`llvm.getelementptr`+`llvm.store`
/// triple per element, fully unrolled — every dimension is a compile-time
/// constant, the same posture `copy_nested_array` already takes for a nested
/// array literal. `src`'s own representation is checked once (`is_mem_ref`)
/// rather than re-derived per element — mirrors `lower_array_load`/
/// `lower_array_store`'s own runtime dispatch on the very same predicate.
/// `elem_mlir_ty` is the leaf's own scalar MLIR type (needed only for the
/// pointer branch's own `llvm.load` result type — `memref.load` infers it
/// from the memref's own element type instead).
fn copy_array_into_llvm_field<'c>(
    ctx: &LowerCtx<'c, '_>,
    block: &Block<'c>,
    src: Value<'c, 'c>,
    dims: &[i64],
    field_ptr: Value<'c, 'c>,
    field_llvm_ty: Type<'c>,
    elem_mlir_ty: Type<'c>,
) {
    let src_is_mem_ref = src.r#type().is_mem_ref();
    fn walk<'c>(
        ctx: &LowerCtx<'c, '_>,
        block: &Block<'c>,
        src: Value<'c, 'c>,
        src_is_mem_ref: bool,
        remaining: &[i64],
        src_idx: &mut Vec<Value<'c, 'c>>,
        gep_idx: &mut Vec<i64>,
        field_ptr: Value<'c, 'c>,
        field_llvm_ty: Type<'c>,
        elem_mlir_ty: Type<'c>,
    ) {
        let Some((&dim, rest)) = remaining.split_first() else {
            let location = Location::unknown(ctx.context);
            let scalar: Value = if src_is_mem_ref {
                block
                    .append_operation(memref::load(src, src_idx, location))
                    .result(0)
                    .unwrap()
                    .into()
            } else {
                let src_ptr = gep(ctx, block, src, gep_idx, field_llvm_ty);
                block
                    .append_operation(llvm::load(
                        ctx.context,
                        src_ptr,
                        elem_mlir_ty,
                        location,
                        LoadStoreOptions::new(),
                    ))
                    .result(0)
                    .unwrap()
                    .into()
            };
            let dst_ptr = gep(ctx, block, field_ptr, gep_idx, field_llvm_ty);
            block.append_operation(llvm::store(
                ctx.context,
                scalar,
                dst_ptr,
                location,
                LoadStoreOptions::new(),
            ));
            return;
        };
        for i in 0..dim {
            src_idx.push(const_index(ctx, block, i));
            gep_idx.push(i);
            walk(
                ctx,
                block,
                src,
                src_is_mem_ref,
                rest,
                src_idx,
                gep_idx,
                field_ptr,
                field_llvm_ty,
                elem_mlir_ty,
            );
            src_idx.pop();
            gep_idx.pop();
        }
    }
    // Leading `0`: stay within this one array instance (see `gep`'s own doc
    // comment) -- every subsequent index is a real dimension of `dims`. The
    // pointer branch reuses this exact same index list to address `src`
    // too, since `src`'s own inline `!llvm.array` shares the identical
    // shape/layout as the destination field (same cleave `Ty` on both
    // sides).
    let mut gep_idx = vec![0i64];
    walk(
        ctx,
        block,
        src,
        src_is_mem_ref,
        dims,
        &mut Vec::new(),
        &mut gep_idx,
        field_ptr,
        field_llvm_ty,
        elem_mlir_ty,
    );
}

fn alloc_array<'c>(
    ctx: &LowerCtx<'c, '_>,
    block: &Block<'c>,
    memref_ty: MemRefType<'c>,
) -> Value<'c, 'c> {
    let location = Location::unknown(ctx.context);
    let op = block.append_operation(memref::alloc(
        ctx.context,
        memref_ty,
        &[],
        &[],
        None,
        location,
    ));
    op.result(0).unwrap().into()
}

fn const_index<'c>(ctx: &LowerCtx<'c, '_>, block: &Block<'c>, n: i64) -> Value<'c, 'c> {
    let location = Location::unknown(ctx.context);
    let attribute = IntegerAttribute::new(Type::index(ctx.context), n).into();
    let op = block.append_operation(arith::constant(ctx.context, attribute, location));
    op.result(0).unwrap().into()
}

fn to_index<'c>(ctx: &LowerCtx<'c, '_>, block: &Block<'c>, value: Value<'c, 'c>) -> Value<'c, 'c> {
    let location = Location::unknown(ctx.context);
    let op = block.append_operation(arith::index_cast(value, Type::index(ctx.context), location));
    op.result(0).unwrap().into()
}

/// Copies every scalar element of `src` (a flat memref of shape `dims`,
/// itself one nested-array "row" — see `lower_array_construct`'s own doc
/// comment) into `dst` at flat position `dst_prefix ++ idx`, one `memref.
/// load`+`memref.store` pair per element, every index fully unrolled (every
/// dimension here is a compile-time constant, cleave has no dynamically-
/// sized arrays). Correctness-first, not throughput-first — a `memref.
/// subview`+`memref.copy` bulk version is a possible later optimization, not
/// attempted here.
fn copy_nested_array<'c>(
    ctx: &LowerCtx<'c, '_>,
    block: &Block<'c>,
    src: Value<'c, 'c>,
    dims: &[i64],
    dst: Value<'c, 'c>,
    dst_prefix: &[Value<'c, 'c>],
) {
    fn walk<'c>(
        ctx: &LowerCtx<'c, '_>,
        block: &Block<'c>,
        src: Value<'c, 'c>,
        remaining: &[i64],
        idx_acc: &mut Vec<Value<'c, 'c>>,
        dst: Value<'c, 'c>,
        dst_prefix: &[Value<'c, 'c>],
    ) {
        let Some((&dim, rest)) = remaining.split_first() else {
            let location = Location::unknown(ctx.context);
            let load_op = block.append_operation(memref::load(src, idx_acc, location));
            let scalar: Value = load_op.result(0).unwrap().into();
            let mut dst_indices = dst_prefix.to_vec();
            dst_indices.extend(idx_acc.iter().copied());
            block.append_operation(memref::store(scalar, dst, &dst_indices, location));
            return;
        };
        for i in 0..dim {
            idx_acc.push(const_index(ctx, block, i));
            walk(ctx, block, src, rest, idx_acc, dst, dst_prefix);
            idx_acc.pop();
        }
    }
    walk(ctx, block, src, dims, &mut Vec::new(), dst, dst_prefix);
}

/// The *entire* hardcoded-in-Rust surface for primitive operations: builds
/// one MLIR operation, generically, from `op` (the real dialect-qualified
/// name, e.g. `arith.addi` — reconstructed in `cps.rs::convert_expr` from a
/// reserved `mlir::dialect::op(...)` call's own path) and `attrs`
/// (`ExprKind::Call::mlir_attrs`, carried through unchanged: attribute name
/// -> raw MLIR attribute text, parsed here via `Attribute::parse`). No
/// per-op-name Rust knowledge anywhere — matches `doc/hld.md`'s own "one
/// generic 'emit this named MLIR op' primitive" goal directly, with three
/// deliberate exceptions, each checked first, before the generic path:
/// `tensor.extract`'s own variadic-index-array form (see below), and
/// `linalg.matmul`/`linalg.transpose` (`build_matmul_no_seed`'s own doc
/// comment — both need a real payload region, which the fully generic
/// builder never attaches, and matmul specifically needs one shaped a
/// particular way its own verifier requires).
///
/// Positional arguments need *some* expected MLIR type to materialize a
/// bare literal against (`mlir::arith::addi(0, x)`) — since there's no
/// per-op signature left to consult, this uses the first already-typed
/// (`CVal::Var`) sibling operand's own MLIR type (`Value::r#type`, always
/// available once lowered), falling back to the op's own declared result
/// type for the (rarer) all-literal case.
/// `tensor.extract`'s own variadic-index-array form — see `lower_raw_mlir_
/// op`'s own doc comment for why this exists. `idx_val`'s own static length
/// (`ShapedTypeLike::dim_size`, always static — cleave has no dynamically-
/// sized arrays) is read directly off its already-lowered `memref` type,
/// each element loaded out via an ordinary `memref.load` (constant index,
/// fully unrolled, same shape `flatten_memref_elements` already uses) and
/// cast to `index` (`to_index`, the type every real `tensor.extract`/
/// `tensor.insert` index operand needs, `i32`'s own array element type
/// otherwise mismatching it).
fn lower_tensor_extract_spread<'c>(
    ctx: &LowerCtx<'c, '_>,
    block: &Block<'c>,
    tensor_val: Value<'c, 'c>,
    idx_array_val: Value<'c, 'c>,
    result_ty: Type<'c>,
) -> Value<'c, 'c> {
    let context = ctx.context;
    let location = Location::unknown(context);
    let memref_ty = MemRefType::try_from(idx_array_val.r#type()).unwrap_or_else(|e| {
        panic!("MLIR lowering: `tensor.extract`'s own index-array argument isn't a memref: {e}")
    });
    let DimSize::Static(k) = memref_ty.dim_size(0).unwrap_or_else(|e| {
        panic!("MLIR lowering: `tensor.extract`'s own index-array argument has no dimension 0: {e}")
    }) else {
        panic!(
            "MLIR lowering: `tensor.extract`'s own index-array argument must have a static length"
        );
    };
    let mut operands = vec![tensor_val];
    for i in 0..k as i64 {
        let idx = const_index(ctx, block, i);
        let load_op = block.append_operation(memref::load(idx_array_val, &[idx], location));
        let scalar: Value = load_op.result(0).unwrap().into();
        operands.push(to_index(ctx, block, scalar));
    }
    let built = OperationBuilder::new("tensor.extract", location)
        .add_operands(&operands)
        .add_results(&[result_ty])
        .build()
        .unwrap_or_else(|e| panic!("MLIR lowering: failed to build tensor.extract: {e}"));
    block.append_operation(built).result(0).unwrap().into()
}

fn lower_raw_mlir_op<'c>(
    ctx: &LowerCtx<'c, '_>,
    block: &Block<'c>,
    env: &HashMap<CVar, Value<'c, 'c>>,
    op: &str,
    attrs: &[(String, String)],
    args: &[CVal],
    result_ty: Type<'c>,
) -> Value<'c, 'c> {
    // `tensor.extract`'s own real MLIR arity is "one index operand per
    // tensor dimension" — fine for a fixed rank (`Index<Vector<T,N>,T>`'s
    // own `mlir::tensor::extract(v, i0)`, still handled by the fully
    // generic path below, untouched), but a *pack*-generic `Index` impl
    // (`Index<Tensor<T,Dims...>,T>`) only knows its own rank once
    // monomorphized, and needs to pass however many indices that turns out
    // to be — packed into one `idx: [i32; Dims.len()]` array parameter,
    // never spread at the cleave call site (no general pack-expansion
    // syntax exists, deliberately — see `doc/backlog.md`'s own note). This
    // reads that array's own already-known static length straight off its
    // already-lowered `memref` type and splices its elements in as the
    // real op's own separate index operands — the same shape `lower_
    // tagged_struct_construct` already uses for `tensor.from_elements`
    // (read every element out of an array value, feed them to an op
    // builder), just applied to `tensor.extract` too. Only fires when the
    // call's own second argument is genuinely array-typed (`i0: i32` isn't
    // a memref, so `Index<Vector<T,N>,T>`'s own existing call shape falls
    // straight through to the generic path below, unchanged).
    if op == "tensor.extract" {
        if let [CVal::Var(base_var), CVal::Var(idx_var)] = args {
            if let (Some(&base_val), Some(&idx_val)) = (env.get(base_var), env.get(idx_var)) {
                if idx_val.r#type().is_mem_ref() {
                    return lower_tensor_extract_spread(ctx, block, base_val, idx_val, result_ty);
                }
            }
        }
    }
    // `linalg.matmul`/`linalg.transpose` — see `build_matmul_no_seed`'s own
    // doc comment for why these two need real, dedicated Rust code (not the
    // generic `linalg.`-prefix path just below, and not the *named* ops
    // that path used to build): both now build their own seed-free
    // destination internally (`tensor.empty()`, from `result_ty` alone),
    // so neither reaches here with a third (`init`) argument any more.
    if op == "linalg.matmul" {
        return build_matmul_no_seed(ctx, block, env, args, result_ty);
    }
    if op == "linalg.transpose" {
        return build_transpose_no_seed(ctx, block, env, args, attrs, result_ty);
    }
    let context = ctx.context;
    let operand_ty = args
        .iter()
        .find_map(|a| match a {
            CVal::Var(v) => env.get(v).map(ValueLike::r#type),
            _ => None,
        })
        .unwrap_or(result_ty);
    let arg_values: Vec<Value> = args
        .iter()
        .map(|a| lower_cval(context, block, env, a, operand_ty))
        .collect();
    let parsed_attrs: Vec<_> = attrs
        .iter()
        .map(|(name, text)| {
            let attribute = Attribute::parse(context, text).unwrap_or_else(|| {
                panic!("MLIR lowering: invalid MLIR attribute text `{text}` for `{name}` on `{op}`")
            });
            (Identifier::new(context, name), attribute)
        })
        .collect();
    let location = Location::unknown(context);
    let builder = OperationBuilder::new(op, location)
        .add_operands(&arg_values)
        .add_attributes(&parsed_attrs)
        .add_results(&[result_ty]);
    let built = builder
        .build()
        .unwrap_or_else(|e| panic!("MLIR lowering: failed to build op `{op}`: {e}"));
    let result_op = block.append_operation(built);
    result_op.result(0).unwrap().into()
}

/// Builds `A @ B` (`Tensor<T,N,M> x Tensor<T,M,K> -> Tensor<T,N,K>`) as a
/// genuinely seed-free `linalg.generic` — no `Ring::zero()`, no `linalg.
/// fill`, no physical zero-initialization of the destination at all.
///
/// **Why this needs real, dedicated Rust code, not the generic `mlir::...`
/// path every other primitive uses**: `linalg.matmul` (the *named* op this
/// used to build, via a shared, dialect-family-wide payload-region builder)
/// is documented as `C := A@B + C` — real BLAS GEMM accumulate semantics,
/// confirmed the hard way (`stdlib/linalg/matrix.cleave`'s own `matmul`
/// impl doc comment has the full story of the correctness bug a stale,
/// uninitialized destination caused the *first* time this code tried
/// skipping the seed) — and its own verifier *rejects* any payload region
/// shaped differently from the canonical multiply-then-add (confirmed
/// directly against this toolchain: `mlir-opt` on a hand-built `arith.
/// select`-based alternative fails with "expected add/mul op in the
/// body"). There is no way to keep the *named* op and still avoid a real,
/// physical seed. `linalg.generic` — the fully generic structured-op form,
/// with explicit indexing maps built by hand below — has no such
/// restriction.
///
/// **The trick, verified directly against this toolchain (`mlir-opt`, a
/// scratch probe carried through `--one-shot-bufferize --buffer-
/// deallocation-pipeline --convert-linalg-to-affine-loops --affine-super-
/// vectorize --lower-affine --convert-vector-to-llvm --convert-to-llvm`)
/// before being written here**: the payload region reads `linalg.index 2`
/// (the current position along the contracted `k` dimension — matmul's own
/// reduction dim, always the *last* of its three iteration dims by MLIR's
/// own named-op convention, mirrored here in the hand-built indexing maps)
/// and `arith.select`s between `a*b` (at `k == 0`, ignoring `outs`'s own
/// value entirely) and `outs + a*b` (at `k > 0`, accumulating as usual).
/// Mathematically identical to the zero-seeded version in exact arithmetic
/// (`0.0 + x == x`, exact under IEEE-754 for any finite `x`) — and confirmed
/// structurally to survive vectorization unchanged: the `select` lowers to
/// an ordinary masked `vector.select`/`llvm.select`, no different from any
/// other elementwise op already in the loop body. `outs`'s own seed
/// (`tensor.empty()`, genuinely uninitialized) never actually reaches the
/// final result: the discarded `k == 0` branch's own `outs + a*b` *is*
/// computed (IEEE-754 float arithmetic on garbage bits is always well-
/// defined, never UB, just an unspecified *value* — unlike, say, reading
/// garbage as a pointer) but never *selected*. This is the exact bug this
/// impl's own history already ran into once (see the doc comment this
/// replaced, in `stdlib/linalg/matrix.cleave`) — the difference is that bug
/// came from a region that read `outs` *unconditionally*; this one only
/// ever reads it on the branch it never picks.
fn build_matmul_no_seed<'c>(
    ctx: &LowerCtx<'c, '_>,
    block: &Block<'c>,
    env: &HashMap<CVar, Value<'c, 'c>>,
    args: &[CVal],
    result_ty: Type<'c>,
) -> Value<'c, 'c> {
    let context = ctx.context;
    let location = Location::unknown(context);
    let [a_arg, b_arg] = args else {
        panic!(
            "MLIR lowering: `mlir::linalg::matmul` needs exactly two operands (`a`, `b`), got {}",
            args.len()
        );
    };
    // `expected_type` only matters for a bare-literal `CVal` (`lower_cval`'s
    // own doc comment) — `a`/`b` are always already-typed tensor values here,
    // so what's passed is irrelevant; `result_ty` is simply whatever is
    // already on hand.
    let a = lower_cval(context, block, env, a_arg, result_ty);
    let b = lower_cval(context, block, env, b_arg, result_ty);
    let elem_ty = RankedTensorType::try_from(result_ty)
        .unwrap_or_else(|e| {
            panic!("MLIR lowering: matmul's own result must be a ranked tensor: {e}")
        })
        .element();
    let init = block
        .append_operation(
            OperationBuilder::new("tensor.empty", location)
                .add_results(&[result_ty])
                .build()
                .unwrap_or_else(|e| panic!("MLIR lowering: failed to build tensor.empty: {e}")),
        )
        .result(0)
        .unwrap()
        .into();

    let index_ty = Type::index(context);
    let payload = Block::new(&[(elem_ty, location), (elem_ty, location), (elem_ty, location)]);
    let av: Value = payload.argument(0).unwrap().into();
    let bv: Value = payload.argument(1).unwrap().into();
    let cv: Value = payload.argument(2).unwrap().into();
    let k: Value = payload
        .append_operation(
            OperationBuilder::new("linalg.index", location)
                .add_attributes(&[(
                    Identifier::new(context, "dim"),
                    IntegerAttribute::new(IntegerType::new(context, 64).into(), 2).into(),
                )])
                .add_results(&[index_ty])
                .build()
                .unwrap_or_else(|e| panic!("MLIR lowering: failed to build linalg.index: {e}")),
        )
        .result(0)
        .unwrap()
        .into();
    let zero_index: Value = payload
        .append_operation(arith::constant(
            context,
            IntegerAttribute::new(index_ty, 0).into(),
            location,
        ))
        .result(0)
        .unwrap()
        .into();
    let is_first: Value = payload
        .append_operation(
            OperationBuilder::new("arith.cmpi", location)
                .add_attributes(&[(
                    Identifier::new(context, "predicate"),
                    IntegerAttribute::new(IntegerType::new(context, 64).into(), 0).into(),
                )])
                .add_operands(&[k, zero_index])
                .add_results(&[IntegerType::new(context, 1).into()])
                .build()
                .unwrap_or_else(|e| panic!("MLIR lowering: failed to build arith.cmpi: {e}")),
        )
        .result(0)
        .unwrap()
        .into();
    // Select the *addend* (`0.0` at `k == 0`, `cv` otherwise), not the
    // final result -- found the hard way, not the first thing tried: an
    // earlier version of this region computed `prod = a*b` once and
    // selected between `prod` (k==0) and `cv + prod` (k>0) directly. That
    // shape is mathematically identical but gives `prod` *two* real uses
    // (the select's own true-branch, and the add) -- confirmed directly by
    // disassembling this project's own real, compiled `examples/mnist-
    // interop` kernel (`llvm-objdump`, not guessed from IR alone): only 14
    // `vfmadd*ps` instructions in the whole kernel against 151 separate
    // `vmulps`/`vaddps` pairs, meaning LLVM's own backend almost never
    // fused the multiply-accumulate here at all -- correctly so: fusing
    // `cv + prod` into an FMA when `prod` is *also* needed bare elsewhere
    // would need computing the multiply a second time anyway, no actual
    // win. Selecting the addend instead keeps the multiply's result used
    // in exactly one place (this add), which is what actually lets `--
    // mark_mulf_addf_contract`'s own `fastmath<contract>` stamp turn into a
    // *real* fused multiply-add at the instruction-selection level --
    // confirmed directly on this same probe shape (`mlir-opt`, `--convert-
    // to-llvm`): `llvm.select` on the addend, then `llvm.fmul`/`llvm.fadd`
    // with `fastmathFlags = #llvm.fastmath<contract>}` immediately
    // afterward, the multiply feeding the add and nothing else.
    let zero_elem: Value = payload
        .append_operation(arith::constant(
            context,
            FloatAttribute::new(context, elem_ty, 0.0).into(),
            location,
        ))
        .result(0)
        .unwrap()
        .into();
    let addend: Value = payload
        .append_operation(
            OperationBuilder::new("arith.select", location)
                .add_operands(&[is_first, zero_elem, cv])
                .add_results(&[elem_ty])
                .build()
                .unwrap_or_else(|e| panic!("MLIR lowering: failed to build arith.select: {e}")),
        )
        .result(0)
        .unwrap()
        .into();
    let prod: Value = payload
        .append_operation(
            OperationBuilder::new("arith.mulf", location)
                .add_operands(&[av, bv])
                .add_results(&[elem_ty])
                .build()
                .unwrap(),
        )
        .result(0)
        .unwrap()
        .into();
    let sum: Value = payload
        .append_operation(
            OperationBuilder::new("arith.addf", location)
                .add_operands(&[addend, prod])
                .add_results(&[elem_ty])
                .build()
                .unwrap(),
        )
        .result(0)
        .unwrap()
        .into();
    payload.append_operation(
        OperationBuilder::new("linalg.yield", location)
            .add_operands(&[sum])
            .build()
            .unwrap(),
    );
    let region = Region::new();
    region.append_block(payload);

    let indexing_maps = Attribute::parse(
        context,
        "[affine_map<(i,j,k) -> (i,k)>, affine_map<(i,j,k) -> (k,j)>, affine_map<(i,j,k) -> (i,j)>]",
    )
    .unwrap_or_else(|| panic!("MLIR lowering: failed to parse matmul's own indexing_maps"));
    let iterator_types = Attribute::parse(
        context,
        "[#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>]",
    )
    .unwrap_or_else(|| panic!("MLIR lowering: failed to parse matmul's own iterator_types"));
    let built = OperationBuilder::new("linalg.generic", location)
        .add_operands(&[a, b, init])
        .add_attributes(&[
            (Identifier::new(context, "indexing_maps"), indexing_maps),
            (Identifier::new(context, "iterator_types"), iterator_types),
            (
                Identifier::new(context, "operandSegmentSizes"),
                DenseI32ArrayAttribute::new(context, &[2, 1]).into(),
            ),
        ])
        .add_regions_vec(vec![region])
        .add_results(&[result_ty])
        .build()
        .unwrap_or_else(|e| panic!("MLIR lowering: failed to build matmul's own linalg.generic: {e}"));
    block.append_operation(built).result(0).unwrap().into()
}

/// Builds `A^T` as a genuinely seed-free `linalg.transpose` — no `Ring::
/// zero()`, no physical zero-initialization of the destination at all.
///
/// Unlike `matmul` (`build_matmul_no_seed`'s own doc comment), `linalg.
/// transpose` has no reduction dimension whatsoever — every output element
/// is touched *exactly once* (a pure permutation, not a contraction) — so
/// the old shared payload-region builder's `out = out + in` body was never
/// actually *needed* here at all, only *tolerated*: with a genuinely zero-
/// filled `out`, `0 + in == in`, the same answer a plain `linalg.yield %av`
/// (never reading `out` at all) gives directly, with no seed required.
/// Confirmed directly, not assumed: a hand-built `linalg.transpose` with a
/// pure-yield region round-trips cleanly through `mlir-opt` (verifies, and
/// pretty-prints back to the same sugared `linalg.transpose ins(...) outs
/// (...) permutation = [...]` form a `Ring::zero()`-seeded one already did)
/// — unlike `linalg.matmul`, this named op's own verifier does not require
/// the canonical multiply-accumulate shape.
fn build_transpose_no_seed<'c>(
    ctx: &LowerCtx<'c, '_>,
    block: &Block<'c>,
    env: &HashMap<CVar, Value<'c, 'c>>,
    args: &[CVal],
    attrs: &[(String, String)],
    result_ty: Type<'c>,
) -> Value<'c, 'c> {
    let context = ctx.context;
    let location = Location::unknown(context);
    let [a_arg] = args else {
        panic!(
            "MLIR lowering: `mlir::linalg::transpose` needs exactly one operand (`a`), got {}",
            args.len()
        );
    };
    let a = lower_cval(context, block, env, a_arg, result_ty);
    let elem_ty = RankedTensorType::try_from(result_ty)
        .unwrap_or_else(|e| {
            panic!("MLIR lowering: transpose's own result must be a ranked tensor: {e}")
        })
        .element();
    let init = block
        .append_operation(
            OperationBuilder::new("tensor.empty", location)
                .add_results(&[result_ty])
                .build()
                .unwrap_or_else(|e| panic!("MLIR lowering: failed to build tensor.empty: {e}")),
        )
        .result(0)
        .unwrap()
        .into();
    let parsed_attrs: Vec<_> = attrs
        .iter()
        .map(|(name, text)| {
            let attribute = Attribute::parse(context, text).unwrap_or_else(|| {
                panic!("MLIR lowering: invalid MLIR attribute text `{text}` for `{name}` on `linalg.transpose`")
            });
            (Identifier::new(context, name), attribute)
        })
        .collect();

    let payload = Block::new(&[(elem_ty, location), (elem_ty, location)]);
    let av: Value = payload.argument(0).unwrap().into();
    payload.append_operation(
        OperationBuilder::new("linalg.yield", location)
            .add_operands(&[av])
            .build()
            .unwrap(),
    );
    let region = Region::new();
    region.append_block(payload);

    let built = OperationBuilder::new("linalg.transpose", location)
        .add_operands(&[a, init])
        .add_attributes(&parsed_attrs)
        .add_regions_vec(vec![region])
        .add_results(&[result_ty])
        .build()
        .unwrap_or_else(|e| panic!("MLIR lowering: failed to build linalg.transpose: {e}"));
    block.append_operation(built).result(0).unwrap().into()
}

/// Emits a `func.func private @symbol(param_types) -> result_ty` declaration
/// (an *empty* region — zero blocks — is what makes it a declaration rather
/// than a definition; confirmed against melior's own `compile_external_
/// function` test) the first time `symbol` is seen, and no-ops on every
/// later call site for the same symbol.
fn ensure_extern_declared<'c>(
    ctx: &LowerCtx<'c, '_>,
    symbol: &str,
    param_types: &[Ty],
    results: &[Type<'c>],
) {
    let mut declared = ctx.declared_externs.borrow_mut();
    if !declared.insert(symbol.to_string()) {
        return;
    }
    let context = ctx.context;
    // A `Ty::Array` param becomes *two* real scalar params here (`!llvm.ptr`,
    // `i64`), never a `memref` — see `array_ptr_and_len`'s own doc comment
    // for why: MLIR's default `convert-to-llvm` conversion turns a `memref`
    // crossing a `func.call` boundary into a descriptor *struct*, not a bare
    // pointer, which no hand-written `cleave-rt` extern fn could plausibly
    // match. The *declared* signature here must agree with what the actual
    // call site (`lower_prim_op`'s own `PrimOp::Extern` arm) really passes.
    let param_mlir: Vec<Type> = param_types
        .iter()
        .flat_map(|t| {
            if matches!(t, Ty::Array(..)) {
                vec![
                    llvm::r#type::pointer(context, 0),
                    IntegerType::new(context, 64).into(),
                ]
            } else {
                vec![ty_to_mlir(ctx, t)]
            }
        })
        .collect();
    let location = Location::unknown(context);
    let decl = func::func(
        context,
        StringAttribute::new(context, symbol),
        TypeAttribute::new(FunctionType::new(context, &param_mlir, results).into()),
        Region::new(),
        &[(
            melior::ir::Identifier::new(context, "sym_visibility"),
            StringAttribute::new(context, "private").into(),
        )],
        location,
    );
    ctx.module.body().append_operation(decl);
}

/// `expected_type` covers the cases that need it: a bare literal `CVal`
/// (`CVal::Int`/`Float`/`Bool` carry no width of their own in the CPS IR —
/// only a surrounding `LetPrim`'s own `ty` field, or an intrinsic/extern
/// call's own known operand type, normally supplies one) flowing into a
/// function's own `return`, an `extern`/intrinsic call's own argument list,
/// or an `if`'s own condition.
fn lower_cval<'c>(
    context: &'c Context,
    block: &Block<'c>,
    env: &HashMap<CVar, Value<'c, 'c>>,
    v: &CVal,
    expected_type: Type<'c>,
) -> Value<'c, 'c> {
    match v {
        CVal::Var(var) => *env
            .get(var)
            .unwrap_or_else(|| panic!("MLIR lowering: unbound CPS variable v{var}")),
        CVal::Int(n) => {
            let location = Location::unknown(context);
            let attribute = IntegerAttribute::new(expected_type, *n as i64).into();
            let op = block.append_operation(arith::constant(context, attribute, location));
            op.result(0).unwrap().into()
        }
        CVal::Float(n) => {
            let location = Location::unknown(context);
            let attribute = FloatAttribute::new(context, expected_type, *n).into();
            let op = block.append_operation(arith::constant(context, attribute, location));
            op.result(0).unwrap().into()
        }
        CVal::Bool(b) => {
            let location = Location::unknown(context);
            let attribute = IntegerAttribute::new(expected_type, *b as i64).into();
            let op = block.append_operation(arith::constant(context, attribute, location));
            op.result(0).unwrap().into()
        }
        // `CVal::Unit` reaching *this* function specifically (as opposed to
        // the several other places in this module that already filter it
        // out before ever calling `lower_cval`) means a genuinely never-
        // read placeholder of some *other*, real type — `doc/backlog-done.
        // md`'s own "break value" item: `ExprKind::Loop`'s own conversion
        // (`cps.rs`) seeds its `break_val` carried slot's very first,
        // pre-any-break value with `CVal::Unit` (there's no other type-
        // agnostic "nothing yet" value available at the CPS level for an
        // arbitrary, possibly non-unit carried type) — a real value only
        // ever lands there via an actual `break value;`, so this is only
        // ever the *first* iteration's own unread initial value. `llvm.mlir.
        // undef` is MLIR's own standard "a real value of this type, content
        // deliberately unspecified" primitive — exactly this case, not a
        // hack: works for any `expected_type`, builtin or dialect-specific.
        CVal::Unit => {
            let location = Location::unknown(context);
            block
                .append_operation(llvm::undef(expected_type, location))
                .result(0)
                .unwrap()
                .into()
        }
        other => panic!("MLIR lowering doesn't support this CVal shape yet: {other:?}"),
    }
}
