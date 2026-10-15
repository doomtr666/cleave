//! The scalar types the compiler knows by name: the integer and floating-point widths every backend
//! lowers natively. Their arithmetic is the stdlib's (`stdlib/num`, `#[mlir_type(...)]` impls); the
//! compiler only needs to know which names are scalars, and which of those are floats (a literal
//! `1.5` of that type, a gradient's seed `1.0`, a zero). One list, so a new width (`bf16`,
//! `doc/plan-quantization.md`) is one line here plus its stdlib impls.

/// The floating-point types, narrowest first. `bf16` (bfloat16: f32's exponent, 8 bits of
/// mantissa) and `f16` (IEEE half) are storage and tensor-core types; their arithmetic rounds to
/// their width, as the type says.
pub const FLOAT_TYPES: &[&str] = &["bf16", "f16", "f32", "f64"];

/// The integer types.
pub const INTEGER_TYPES: &[&str] = &["i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64"];

/// Whether `name` is a floating-point type.
pub fn is_float(name: &str) -> bool {
    FLOAT_TYPES.contains(&name)
}

/// Whether `name` is a scalar: an integer, a float, or `bool`.
pub fn is_scalar(name: &str) -> bool {
    is_float(name) || INTEGER_TYPES.contains(&name) || name == "bool"
}
