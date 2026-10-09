//! The compiler's hash maps and sets: `std`'s, with a hasher whose keys are
//! fixed rather than drawn per process (`RandomState`), so they iterate in the
//! same order on every run. A pass that emits code while iterating one (the
//! releases of the values dying at a point, the terms of a derivative) then
//! emits the same code every time: a compilation is reproducible, and two
//! compilers' outputs can be compared.

pub type FixedState = std::hash::BuildHasherDefault<std::collections::hash_map::DefaultHasher>;
pub type HashMap<K, V> = std::collections::HashMap<K, V, FixedState>;
pub type HashSet<T> = std::collections::HashSet<T, FixedState>;
