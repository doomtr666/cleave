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
        ("sub", ConstValue::Int(x), ConstValue::Int(y)) => Some(ConstValue::Int(x.wrapping_sub(y))),
        // Plain, unsigned `u64` division — correct for this function's own
        // const-generic callers (`infer.rs::const_value_from_expr`/
        // `fold_const_expr`): an array size is never negative, so there's no
        // signed-vs-unsigned ambiguity to get wrong there. Division by a
        // *statically zero* divisor returns `None` here (this function's own
        // existing "can't evaluate this" signal, same as an unrecognized
        // operator) rather than panicking — `infer.rs`'s own caller is what
        // turns a zero divisor into a real, located compile error
        // (`TypeErrorKind::ConstDivByZero`, `pending_div_by_zero_checks`),
        // since only it has a source span to point at.
        //
        // Deliberately *not* reused by `egraph.rs::ConstantFold` for
        // opportunistic runtime folding, unlike `add`/`mul`/`sub` — those
        // three give the identical `u64` bit pattern whether the operand is
        // "meant" to be signed or unsigned (two's-complement wrapping
        // arithmetic doesn't care), but division genuinely differs for a
        // negative signed operand, and neither `ConstValue::Int` nor
        // `CleaveLang::Int` carries a width/signedness tag to tell which is
        // meant — folding a real `Ring::div<i32>` here could silently
        // compute the *wrong* answer instead of just missing an
        // optimization. See `ConstantFold::make`'s own `div` handling.
        ("div", ConstValue::Int(x), ConstValue::Int(y)) => (y != 0).then(|| ConstValue::Int(x / y)),
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
    fn sub_subtracts_two_ints() {
        assert_eq!(eval_binop("sub", ConstValue::Int(10), ConstValue::Int(10)), Some(ConstValue::Int(0)));
        assert_eq!(eval_binop("sub", ConstValue::Int(4), ConstValue::Int(3)), Some(ConstValue::Int(1)));
    }

    #[test]
    fn a_bool_operand_is_not_arithmetic() {
        assert_eq!(eval_binop("add", ConstValue::Bool(true), ConstValue::Int(3)), None);
    }

    #[test]
    fn an_unrecognized_operator_is_not_evaluated() {
        assert_eq!(eval_binop("mod", ConstValue::Int(4), ConstValue::Int(3)), None);
    }

    #[test]
    fn div_divides_two_ints_truncating() {
        assert_eq!(eval_binop("div", ConstValue::Int(7), ConstValue::Int(2)), Some(ConstValue::Int(3)));
        assert_eq!(eval_binop("div", ConstValue::Int(4), ConstValue::Int(4)), Some(ConstValue::Int(1)));
    }

    #[test]
    fn div_by_zero_is_not_evaluated_rather_than_panicking() {
        assert_eq!(eval_binop("div", ConstValue::Int(4), ConstValue::Int(0)), None);
    }
}
