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
    /// The `#[mlir(...)]` instruction name — nothing here reads it further
    /// than carrying it into `PrimOp::Intrinsic`; MLIR lowering (a later,
    /// separate stage) is what actually emits it.
    Intrinsic(String),
}

pub struct ConcreteUnit {
    pub name: String,
    pub params: Vec<Param>,
    pub param_types: Vec<Ty>,
    pub result: Ty,
    pub node_types: HashMap<NodeId, Ty>,
    /// This unit's own resolved mangled-callee-name map, exactly as
    /// `monomorphize.rs` already builds it — empty for a concrete impl
    /// method (no specialization involved, see the module's own doc
    /// comment on `call_index` for how those calls resolve instead).
    pub call_names: HashMap<NodeId, String>,
    pub body: UnitBody,
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
                    // Non-generic — a bodyless top-level `fn` is already
                    // rejected by `callgraph::infer_program` itself
                    // (`MissingFnBody`), so `results` would hold an `Err`
                    // here, not `Ok` — `f.body` is always `Some` at this point.
                    let Some(body) = &f.body else { continue };
                    units.push(ConcreteUnit {
                        name: f.name.clone(),
                        params: f.params.clone(),
                        param_types: fn_result.param_types.clone(),
                        result: fn_result.result.clone(),
                        node_types: program_inference.node_types.clone(),
                        call_names: mono.seed_call_names().clone(),
                        body: UnitBody::Real(body.clone()),
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
                            body: UnitBody::Real(mono.body(key).clone()),
                        });
                    }
                }
            }
            ItemKind::Impl(d) if d.generics.is_empty() => {
                // Non-generic algebra impl — mirrors `monomorphize.rs`'s own
                // `dump_concrete_impl`, re-inferring directly (no template
                // needed, it's already fully concrete) rather than reusing
                // that function's own rendering logic.
                let all_targets: Vec<Type> =
                    std::iter::once(d.target.clone()).chain(d.extra_targets.iter().cloned()).collect();
                for f in &d.fns {
                    let mut infer = Infer::new(registry);
                    let Ok(ret) =
                        infer.infer_impl_fn_generic_with_env(&program_inference.global_env, &d.algebra, &d.generics, &all_targets, f, item.span)
                    else {
                        continue; // already reported via --dump-inference-pass
                    };
                    let body = match &f.body {
                        Some(b) => UnitBody::Real(b.clone()),
                        None => {
                            let Some(instr) = f.attrs.iter().find(|a| a.name == "mlir").and_then(|a| a.args.first())
                            else {
                                continue; // MissingIntrinsicAttribute, already reported
                            };
                            UnitBody::Intrinsic(instr.clone())
                        }
                    };
                    let targets_str = infer.target_types.iter().map(Ty::to_string).collect::<Vec<_>>().join(", ");
                    units.push(ConcreteUnit {
                        name: format!("{}::{}<{targets_str}>", d.algebra, f.name),
                        params: f.params.clone(),
                        param_types: infer.param_types.clone(),
                        result: ret,
                        node_types: infer.node_types.clone(),
                        call_names: HashMap::new(),
                        body,
                    });
                }
            }
            ItemKind::Impl(d) => {
                // Generic algebra impl — every specialization actually
                // reached, already built by `monomorphize`.
                for f in &d.fns {
                    let origin = format!("{}::{}", d.algebra, f.name);
                    for key in mono.specializations_of(&origin) {
                        units.push(ConcreteUnit {
                            name: key.clone(),
                            params: mono.params(key).to_vec(),
                            param_types: mono.param_types(key).to_vec(),
                            result: mono.result(key).clone(),
                            node_types: mono.node_types(key).clone(),
                            call_names: mono.call_names(key).clone(),
                            body: UnitBody::Real(mono.body(key).clone()),
                        });
                    }
                }
            }
            _ => {}
        }
    }
    units
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
}

#[derive(Debug, Clone)]
pub enum PrimOp {
    Intrinsic(String),
    Field(String),
    Struct(String, Vec<String>),
    /// `args` are the element values, in order (`ExprKind::ArrayLit`).
    Array,
    /// `args = [value, count]` (`ExprKind::ArrayRepeat`).
    ArrayRepeat,
    /// `args = [array, index]` — always safe to nest (see the module's own
    /// "Arrays" doc comment).
    Load,
    /// `args = [array, index, value]` — a real effect, mutating `array` in
    /// place; its own bound result is unit and never read.
    Store,
}

pub enum CExpr {
    LetPrim { var: CVar, ty: Ty, op: PrimOp, args: Vec<CVal>, cont: Box<CExpr> },
    /// `func(args)` — for a *real* callee, `args`' own last entry is, by
    /// convention, the continuation to invoke with the result. A
    /// `PrimOp::Intrinsic` never appears here — that's a `LetPrim`.
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

pub struct CFunDef {
    pub name: String,
    /// Ordinary parameters, plus — for a *real* function specifically (an
    /// intrinsic is never represented as a `CFunDef` at all, only as
    /// `PrimOp::Intrinsic`) — one trailing continuation parameter.
    pub params: Vec<CVar>,
    pub body: CExpr,
}

pub struct CpsProgram {
    pub funcs: Vec<CFunDef>,
}

// ---------------------------------------------------------------- conversion

struct FreshVars {
    next_var: Cell<u32>,
    next_label: Cell<u32>,
}

impl FreshVars {
    fn new() -> Self {
        FreshVars { next_var: Cell::new(0), next_label: Cell::new(0) }
    }

    fn var(&self) -> CVar {
        let v = self.next_var.get();
        self.next_var.set(v + 1);
        v
    }

    fn label(&self, hint: &str) -> String {
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
    fresh: &'a FreshVars,
}

type CEnv = HashMap<String, CVal>;

/// Converts every `ConcreteUnit` with a real body into one `CFunDef` —
/// intrinsics never get one at all (see `PrimOp::Intrinsic`, produced
/// directly at their own call sites instead).
pub fn convert_program(units: Vec<ConcreteUnit>) -> CpsProgram {
    let call_index = build_call_index(&units);
    let by_name: HashMap<String, ConcreteUnit> = units.into_iter().map(|u| (u.name.clone(), u)).collect();
    let fresh = FreshVars::new();
    let mut funcs = Vec::new();
    for unit in by_name.values() {
        let UnitBody::Real(body) = &unit.body else { continue };
        let ctx = Ctx { units: &by_name, call_index: &call_index, node_types: &unit.node_types, call_names: &unit.call_names, fresh: &fresh };
        let mut env = CEnv::new();
        let mut params = Vec::with_capacity(unit.params.len() + 1);
        for p in &unit.params {
            let v = fresh.var();
            env.insert(p.name.clone(), CVal::Var(v));
            params.push(v);
        }
        let k_ret = fresh.var();
        params.push(k_ret);
        let cexpr = convert_block(body, &env, &ctx, &|v, _env| CExpr::App { func: CVal::Var(k_ret), args: vec![v] });
        funcs.push(CFunDef { name: unit.name.clone(), params, body: cexpr });
    }
    // Deterministic output order — `HashMap` iteration isn't stable.
    funcs.sort_by(|a, b| a.name.cmp(&b.name));
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
            // A chain of nested `Index` nodes (`a[i,j]`'s own Fortran sugar
            // *and* a literal `a[i][j]` both desugar to the exact same
            // `Index(Index(a,i),j)` shape -- indistinguishable after
            // lowering, and, in this language, semantically identical too: a
            // multi-dim array's own type is always `Array(Array(T,C),R)`,
            // never a separate primitive) collapses into *one* combined,
            // multi-index `Store` — see the module's own "Arrays" doc
            // comment for why this matters for correctness, not just
            // efficiency: a write through an *intermediate* single-index
            // `Load` (getting "the row", then storing into that) would only
            // be correct if `Load` on an array-of-arrays element aliased the
            // original storage rather than copying it out — a real, load-
            // bearing representation choice this module never actually
            // makes, so this instead never produces that intermediate `Load`
            // at all: the whole chain resolves to one flat offset in a
            // single effect.
            ExprKind::Index(..) => {
                let (array_expr, index_exprs) = collect_index_chain(target);
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
                                op: PrimOp::Store,
                                args,
                                cont: Box::new(convert_stmts(rest, env.clone(), ctx, k)),
                            }
                        })
                    })
                })
            }
            ExprKind::FieldAccess(..) => {
                panic!("CPS Stage 5 doesn't support a field-mutation assignment target (`s.x = v`) yet -- see the module's own \"Arrays\" doc comment (doc/backlog.md)")
            }
            other => panic!("CPS: unexpected assignment target {other:?}"),
        },
    }
}

fn convert_expr(expr: &Expr, env: &CEnv, ctx: &Ctx, k: &dyn Fn(CVal, &CEnv) -> CExpr) -> CExpr {
    match &expr.kind {
        ExprKind::NumberLit { text, .. } => {
            let ty = &ctx.node_types[&expr.id];
            k(parse_number(text, ty), env)
        }
        ExprKind::BoolLit(b) => k(CVal::Bool(*b), env),
        ExprKind::Path(p) => {
            let name = p.segments.join("::");
            let v = match env.get(&name) {
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
        ExprKind::FieldAccess(base, field) => convert_expr(base, env, ctx, &|base_val, env| {
            let var = ctx.fresh.var();
            CExpr::LetPrim {
                var,
                ty: ctx.node_types[&expr.id].clone(),
                op: PrimOp::Field(field.clone()),
                args: vec![base_val],
                cont: Box::new(k(CVal::Var(var), env)),
            }
        }),
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
        ExprKind::Call(path, _, args) => {
            let name = path.segments.join("::");
            let arg_refs: Vec<&Expr> = args.iter().collect();
            let result_ty = ctx.node_types[&expr.id].clone();
            convert_expr_list(&arg_refs, env, ctx, &|arg_vals, env| {
                let callee = resolve_call(&name, expr, args, ctx);
                emit_call(callee, arg_vals, result_ty.clone(), ctx, env, k)
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
            let mut mutated: Vec<String> = mutated_free_vars(then_branch, &HashSet::new()).into_iter().collect();
            match else_branch {
                Some(eb) => match &**eb {
                    ElseBranch::If(e) => mutated.extend(mutated_free_vars_expr(e, &HashSet::new())),
                    ElseBranch::Block(b) => mutated.extend(mutated_free_vars(b, &HashSet::new())),
                },
                None => {}
            }
            mutated.sort();
            mutated.dedup();
            let carried: Vec<(String, CVar)> = mutated.iter().map(|n| (n.clone(), ctx.fresh.var())).collect();

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

            CExpr::Fix {
                defs: vec![CFunDef { name: join_label, params: join_params, body: join_body }],
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
            let mut mutated: Vec<String> = mutated_free_vars_expr(cond, &HashSet::new()).into_iter().collect();
            mutated.extend(mutated_free_vars(body, &HashSet::new()));
            mutated.sort();
            mutated.dedup();
            let carried: Vec<(String, CVar)> = mutated.iter().map(|n| (n.clone(), ctx.fresh.var())).collect();

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
                defs: vec![CFunDef { name: loop_label.clone(), params, body: check }],
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
                let mut mutated: Vec<String> = mutated_free_vars(body, &shadowed).into_iter().collect();
                mutated.sort();
                let carried: Vec<(String, CVar)> = mutated.iter().map(|n| (n.clone(), ctx.fresh.var())).collect();

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

                CExpr::Fix {
                    defs: vec![CFunDef { name: loop_label.clone(), params, body: cond_check }],
                    body: Box::new(CExpr::App { func: CVal::Label(loop_label), args: init_args }),
                }
            })
        }),
        // Collapses a whole chain of nested `Index` nodes into one combined,
        // multi-index `Load` -- see `StmtKind::Assign`'s own `Index` arm for
        // why (the same reasoning applies to reads, for consistency, even
        // though a read alone has no aliasing hazard either way).
        ExprKind::Index(..) => {
            let (array_expr, index_exprs) = collect_index_chain(expr);
            convert_expr(array_expr, env, ctx, &|array_val, env| {
                convert_expr_list(&index_exprs, env, ctx, &|index_vals, env| {
                    let var = ctx.fresh.var();
                    let mut args = vec![array_val.clone()];
                    args.extend(index_vals);
                    CExpr::LetPrim {
                        var,
                        ty: ctx.node_types[&expr.id].clone(),
                        op: PrimOp::Load,
                        args,
                        cont: Box::new(k(CVal::Var(var), env)),
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
        other => panic!("CPS doesn't support {other:?} yet -- see doc/backlog.md item 3's own staged roadmap"),
    }
}

/// Walks a chain of nested `Index` nodes (`a[i,j]`'s own Fortran-sugar
/// desugaring, or a literal `a[i][j]` — indistinguishable after lowering,
/// see the module's own "Arrays" doc comment) down to the innermost
/// non-`Index` base, returning it alongside every index expression
/// collected along the way, back in the original *source* order (`a[i,j]`
/// -> `(a, [i, j])`, not `[j, i]` — the walk itself runs outside-in, so the
/// collected order is reversed before returning).
fn collect_index_chain(expr: &Expr) -> (&Expr, Vec<&Expr>) {
    let mut indices = Vec::new();
    let mut current = expr;
    while let ExprKind::Index(base, idx) = &current.kind {
        indices.push(idx.as_ref());
        current = base;
    }
    indices.reverse();
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
/// `while`/`for` — a purely syntactic walk (no dataflow fixpoint needed, the
/// structured AST already says everything relevant) — see the module's own
/// "Mutation across control flow" doc comment. A name `let`/`let mut`-bound
/// *inside* `block` itself shadows any same-named outer variable for the
/// rest of that scope and is correctly excluded. An indexed/field assignment
/// target doesn't name a scalar to thread at all (Stage 5, not handled
/// here).
fn mutated_free_vars(block: &Block, shadowed: &HashSet<String>) -> HashSet<String> {
    let mut local_shadowed = shadowed.clone();
    let mut escaping = HashSet::new();
    for stmt in &block.stmts {
        match &stmt.kind {
            StmtKind::Let { name, value, .. } => {
                escaping.extend(mutated_free_vars_expr(value, &local_shadowed));
                local_shadowed.insert(name.clone());
            }
            StmtKind::Assign { target, value } => {
                escaping.extend(mutated_free_vars_expr(value, &local_shadowed));
                if let ExprKind::Path(p) = &target.kind {
                    let name = p.segments.join("::");
                    if !local_shadowed.contains(&name) {
                        escaping.insert(name);
                    }
                }
            }
            StmtKind::Expr(e) => escaping.extend(mutated_free_vars_expr(e, &local_shadowed)),
        }
    }
    if let Some(tail) = &block.tail {
        escaping.extend(mutated_free_vars_expr(tail, &local_shadowed));
    }
    escaping
}

fn mutated_free_vars_expr(expr: &Expr, shadowed: &HashSet<String>) -> HashSet<String> {
    match &expr.kind {
        ExprKind::NumberLit { .. } | ExprKind::ImaginaryLit { .. } | ExprKind::BoolLit(_) | ExprKind::Path(_) => HashSet::new(),
        ExprKind::Call(_, _, args) => args.iter().flat_map(|a| mutated_free_vars_expr(a, shadowed)).collect(),
        ExprKind::FieldAccess(base, _) => mutated_free_vars_expr(base, shadowed),
        ExprKind::MethodCall(base, _, args) => {
            let mut out = mutated_free_vars_expr(base, shadowed);
            out.extend(args.iter().flat_map(|a| mutated_free_vars_expr(a, shadowed)));
            out
        }
        ExprKind::Index(base, idx) => {
            let mut out = mutated_free_vars_expr(base, shadowed);
            out.extend(mutated_free_vars_expr(idx, shadowed));
            out
        }
        ExprKind::ArrayLit(elems) => elems.iter().flat_map(|e| mutated_free_vars_expr(e, shadowed)).collect(),
        ExprKind::ArrayRepeat { value, count } => {
            let mut out = mutated_free_vars_expr(value, shadowed);
            out.extend(mutated_free_vars_expr(count, shadowed));
            out
        }
        ExprKind::StructLit(_, _, fields) => fields.iter().flat_map(|(_, v)| mutated_free_vars_expr(v, shadowed)).collect(),
        ExprKind::If { cond, then_branch, else_branch } => {
            let mut out = mutated_free_vars_expr(cond, shadowed);
            out.extend(mutated_free_vars(then_branch, shadowed));
            if let Some(eb) = else_branch {
                match &**eb {
                    ElseBranch::If(e) => out.extend(mutated_free_vars_expr(e, shadowed)),
                    ElseBranch::Block(b) => out.extend(mutated_free_vars(b, shadowed)),
                }
            }
            out
        }
        ExprKind::While { cond, body } => {
            let mut out = mutated_free_vars_expr(cond, shadowed);
            out.extend(mutated_free_vars(body, shadowed));
            out
        }
        ExprKind::For { var, start, end, body } => {
            let mut out = mutated_free_vars_expr(start, shadowed);
            out.extend(mutated_free_vars_expr(end, shadowed));
            let mut inner = shadowed.clone();
            inner.insert(var.clone());
            out.extend(mutated_free_vars(body, &inner));
            out
        }
        ExprKind::Block(b) => mutated_free_vars(b, shadowed),
        // A lambda's own body isn't walked here -- lambdas aren't converted
        // at all yet (closure conversion is a separate, later pass this
        // module doesn't implement, see its own doc comment), so nothing
        // downstream would ever read a mutation found inside one anyway.
        ExprKind::Lambda { .. } => HashSet::new(),
    }
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
/// bound checks: an intrinsic becomes a straight-line `LetPrim` (Appel's
/// PRIMOP), a real callee becomes a synthesized continuation plus a tail
/// `App` with it appended (Appel's APP) — see the module's own doc comment.
/// A call never itself mutates `env` (only `Assign` does) — `k` is always
/// invoked with the very same `env` the call started with.
fn emit_call(unit_name: &str, arg_vals: Vec<CVal>, result_ty: Ty, ctx: &Ctx, env: &CEnv, k: &dyn Fn(CVal, &CEnv) -> CExpr) -> CExpr {
    let unit = &ctx.units[unit_name];
    match &unit.body {
        UnitBody::Intrinsic(instr) => {
            let var = ctx.fresh.var();
            CExpr::LetPrim {
                var,
                ty: result_ty,
                op: PrimOp::Intrinsic(instr.clone()),
                args: arg_vals,
                cont: Box::new(k(CVal::Var(var), env)),
            }
        }
        UnitBody::Real(_) => {
            let result_var = ctx.fresh.var();
            let k_label = ctx.fresh.label("k");
            let mut call_args = arg_vals;
            call_args.push(CVal::Label(k_label.clone()));
            CExpr::Fix {
                defs: vec![CFunDef { name: k_label, params: vec![result_var], body: k(CVal::Var(result_var), env) }],
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
fn resolve_call<'a>(name: &str, call: &Expr, args: &[Expr], ctx: &Ctx<'a>) -> &'a str {
    if let Some(mangled) = ctx.call_names.get(&call.id) {
        return ctx.units.get_key_value(mangled.as_str()).map(|(k, _)| k.as_str()).unwrap_or_else(|| {
            panic!("CPS: call_names resolved `{name}` to `{mangled}`, but no such unit exists")
        });
    }
    if let Some((k, _)) = ctx.units.get_key_value(name) {
        return k.as_str();
    }
    let arg_tys: Vec<String> = args.iter().map(|a| ctx.node_types[&a.id].to_string()).collect();
    let ret_ty = ctx.node_types[&call.id].to_string();
    let key = (name.to_string(), arg_tys, ret_ty);
    match ctx.call_index.get(&key) {
        Some(unit_name) => ctx.units.get_key_value(unit_name.as_str()).map(|(k, _)| k.as_str()).unwrap(),
        None => panic!("CPS: could not resolve call to `{name}` ({key:?})"),
    }
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

// ---------------------------------------------------------------- rendering

pub fn dump_cps_program(program: &CpsProgram) -> String {
    let mut out = String::new();
    for (i, f) in program.funcs.iter().enumerate() {
        if i > 0 {
            out.push('\n');
        }
        let params = f.params.iter().map(|v| format!("v{v}")).collect::<Vec<_>>().join(" ");
        let _ = writeln!(out, "(fn {} ({params})", f.name);
        dump_cexpr(&mut out, &f.body, 1);
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
    }
}

fn dump_cexpr(out: &mut String, expr: &CExpr, depth: usize) {
    match expr {
        CExpr::LetPrim { var, ty, op, args, cont } => {
            indent(out, depth);
            let op_str = match op {
                PrimOp::Intrinsic(name) => name.clone(),
                PrimOp::Field(name) => format!("field.{name}"),
                PrimOp::Struct(name, fields) => format!("struct.{name}[{}]", fields.join(",")),
                PrimOp::Array => "array".to_string(),
                PrimOp::ArrayRepeat => "array-repeat".to_string(),
                PrimOp::Load => "load".to_string(),
                PrimOp::Store => "store".to_string(),
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
