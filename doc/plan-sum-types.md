# Plan: enums with data, and `match`

Status: steps 1-3 done (2026-10-10, `backlog-done.md`: "Enums with data and `match`"). Later items open.

## Why

No way to say "absent" or "one of these" today: `HashMap::lookup` needs the key present, an
intersection can't return "nothing, a point or a segment". Sum types, Rust-style, limited at first.

## The language, first version

```
enum Option<T> { None, Some(T) }
enum Hit { Miss, Point(f64, f64), Segment(f64, f64, f64, f64) }

let o = Some(3);            // a variant is a constructor, generic like a fn
let n: Option<i32> = None;  // a variant without data: a bare name
match o {
    Some(x) => x + 1,       // a variant pattern binds its data, `_` ignores one
    None => 0,
}
```

- Variants are reached unqualified (`Some`) or qualified (`Option::Some`); two enums of the program
  can't share a variant name.
- `match` arms: a variant pattern (`Some(x)`, `Point(_, y)`, `None`) or `_`. Every variant covered or
  a `_` arm, else an error naming the missing ones; an arm after `_`, or a variant twice, is an error.
  No nested patterns, guards or bindings of the whole value yet.
- `match` is an expression; its arms' values unify.

## Representation: a struct whose inactive fields are zero

An enum is lowered, right after the program's crates are merged (`driver.rs`), to:
- a struct with a `tag: i32` and one field per piece of data of every variant (`Some.0`,
  `Segment.3`...): a sum of fields, not a union;
- one generic constructor `fn` per variant, building the struct with its tag and its own fields, every
  other field left **all zero** (the struct is flagged so a literal may omit fields: `zero_fill`);
- `match e { .. }` becomes `{ let <match#n> = e; if <match#n>.tag == k { let x = <match#n>.Some.0;
  .. } else if .. }`, the last arm the final `else`.

Everything after sees structs, fields and `if`s, which every pass already handles: light or heavy
classification, refcounting, aliasing, regions, the e-graph. The release cascade needs no switch on
the tag: an inactive field is zero (a null pointer, a null tensor descriptor), which releases nothing,
the convention `Buffer<T>`'s empty slots already rely on.

What lowering adds: zeroing the omitted fields of a `zero_fill` construction (a heavy struct's block,
a light struct's aggregate), and tolerating a null pointer where a cascade or a retain could now meet
one (a heavy struct inside an inactive variant).

## Steps

1. Grammar and AST (`enum_decl`, `match_expr`), the desugaring pass with its errors (exhaustiveness,
   unknown variant, duplicate variant names), `zero_fill` construction. Tests: `Option<i32>`, an enum
   with several payloads, `match` as an expression, the errors.
2. Elements that are refcounted: `Option` of a heavy struct, of a light struct, of a tensor; leak tests
   and `CLEAVE_DEBUG_POOL` runs.
3. `Option<T>` in `stdlib/core`, `HashMap::get` returning `Option<V>`.

Later: nested patterns, guards, a union layout (overlapping fields), the null-pointer niche
(`Option<HeavyStruct>` as a nullable pointer).
