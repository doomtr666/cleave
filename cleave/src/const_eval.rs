//! A small, standalone compile-time constant evaluator — pure arithmetic on
//! `ConstValue`, no dependency on `Infer`/`Ty`/`Expr`. Deliberately isolated
//! from `infer.rs`'s own type-inference glue (which decides *when* an
//! expression is worth evaluating, e.g. `const_value_from_expr`): this module
//! only ever answers "given two already-resolved constants and an operator
//! name, what's the result", so it stays reusable wherever constant folding
//! is needed later (e.g. actual codegen), not just from `infer.rs`.
//!
//! Starting operator set is deliberately narrow (`add`/`mul` — the names
//! `+`/`*` desugar to, see `ast.rs`'s own `Call` doc comment) and meant to
//! grow incrementally: adding `"sub"` or another op is a one-line addition
//! to `eval_binop`'s match, no change to its signature or callers.

use crate::infer::ConstValue;

/// Evaluates `a <op> b` where `op` is one of this project's own desugared
/// operator names (`"add"`, `"mul"`, ...). Returns `None` for an operator
/// this module doesn't know yet, or for any operand shape that operator
/// isn't defined over (e.g. `Bool` for arithmetic) — permissive by omission,
/// same posture `const_value_from_expr`'s own caller already falls back on
/// for "not evaluated yet".
pub fn eval_binop(op: &str, a: ConstValue, b: ConstValue) -> Option<ConstValue> {
    match (op, a, b) {
        ("add", ConstValue::Int(x), ConstValue::Int(y)) => Some(ConstValue::Int(x.wrapping_add(y))),
        ("mul", ConstValue::Int(x), ConstValue::Int(y)) => Some(ConstValue::Int(x.wrapping_mul(y))),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_sums_two_ints() {
        assert_eq!(eval_binop("add", ConstValue::Int(4), ConstValue::Int(3)), Some(ConstValue::Int(7)));
    }

    #[test]
    fn mul_multiplies_two_ints() {
        assert_eq!(eval_binop("mul", ConstValue::Int(4), ConstValue::Int(3)), Some(ConstValue::Int(12)));
    }

    #[test]
    fn a_bool_operand_is_not_arithmetic() {
        assert_eq!(eval_binop("add", ConstValue::Bool(true), ConstValue::Int(3)), None);
    }

    #[test]
    fn an_unrecognized_operator_is_not_evaluated() {
        assert_eq!(eval_binop("sub", ConstValue::Int(4), ConstValue::Int(3)), None);
    }
}
