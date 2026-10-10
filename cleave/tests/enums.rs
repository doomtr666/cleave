//! Enums and `match` (`doc/plan-sum-types.md`, `cleave/src/enums.rs`), run
//! through the real pipeline.

use cleave::pipeline::CodegenOptions;

fn run(src: &str) -> i32 {
    let (program, registry, sources) =
        cleave::run::check_sources(vec![("test.cleave".to_string(), src.to_string())], &[], false)
            .unwrap_or_else(|e| panic!("{}", e.join("\n")));
    let options = CodegenOptions { openmp: false, tasks: false, ..Default::default() };
    cleave::run::run_main(&program, &registry, Some(&sources), &options, &[]).unwrap_or_else(|e| panic!("{}", e.join("\n")))
}

/// The rendered diagnostics of a program that doesn't compile.
fn errors(src: &str) -> Vec<String> {
    match cleave::run::check_sources(vec![("test.cleave".to_string(), src.to_string())], &[], false) {
        Ok(_) => panic!("the program compiled"),
        Err(e) => e,
    }
}

#[test]
fn variants_with_and_without_data_are_built_and_matched() {
    let src = "
        enum Shape { Empty, Circle(f64), Rect(f64, f64) }
        fn area(s: Shape) -> f64 {
            match s {
                Circle(r) => 3.0 * r * r,
                Rect(w, h) => w * h,
                Empty => 0.0,
            }
        }
        fn main() -> i32 {
            let a = area(Circle(2.0)) + area(Shape::Rect(3.0, 4.0)) + area(Empty);
            if a == 24.0 { 1 } else { 0 }
        }
    ";
    assert_eq!(run(src), 1);
}

/// A generic enum: `None`'s type comes from where it's used; `_` arms and
/// ignored data; a `match` inside an arm.
#[test]
fn a_generic_option_is_inferred_from_its_uses() {
    let src = "
        fn half(x: i32) -> Option<i32> { if mod(x, 2) == 0 { Some(x / 2) } else { None } }
        fn unwrap_or<T>(o: Option<T>, d: T) -> T { match o { Some(v) => v, None => d } }
        fn main() -> i32 {
            let a = unwrap_or(half(10), -1);
            let b = unwrap_or(half(7), -1);
            let c = match half(8) {
                Some(q) => match half(q) { Some(r) => r, _ => 100 },
                None => 200,
            };
            let f: Option<f64> = Some(2.5);
            let g = match f { Some(_) => 1, None => 0 };
            if a == 5 and b == -1 and c == 2 and g == 1 { 1 } else { 0 }
        }
    ";
    assert_eq!(run(src), 1);
}

/// Data that is refcounted: a heavy struct, a light struct holding a tensor,
/// a tensor. An inactive variant's fields are zero (a null pointer, a null
/// descriptor), which the release cascade skips.
#[test]
fn variants_holding_structs_and_tensors_round_trip() {
    let src = "
        use nn;
        struct Heavy { t: Tensor<f32, 4, 4>, k: i32 }
        struct Light { t: Tensor<f32, 4, 4>, k: i32 }
        fn bump(mut h: Heavy) { h.k = h.k + 1; }
        fn pick_heavy(m: Tensor<f32, 4, 4>, i: i32) -> Option<Heavy> {
            if mod(i, 2) == 0 { let h = Heavy(t: Scale::scale(m, 2.0), k: i); bump(h); Some(h) } else { None }
        }
        fn pick_light(m: Tensor<f32, 4, 4>, i: i32) -> Option<Light> {
            if mod(i, 3) == 0 { Some(Light(t: Scale::scale(m, 3.0), k: i)) } else { None }
        }
        fn pick_tensor(m: Tensor<f32, 4, 4>, i: i32) -> Option<Tensor<f32, 4, 4>> {
            if mod(i, 2) == 1 { Some(Scale::scale(m, 0.5)) } else { None }
        }
        fn main() -> i32 {
            rand_seed(1);
            let m: Tensor<f32, 4, 4> = Init::xavier();
            let mut s = 0;
            let mut ok = 1;
            for i in 0..6 {
                s = s + match pick_heavy(m, i) { Some(h) => h.k, None => 100 }
                      + match pick_light(m, i) { Some(l) => l.k, None => 1000 };
                match pick_tensor(m, i) {
                    Some(t) => { if t[1, 2] != 0.5 * m[1, 2] { ok = 0; }; },
                    None => {},
                };
            };
            if s == 9 + 300 + 3 + 4000 { ok } else { 0 }
        }
    ";
    assert_eq!(run(src), 1);
}

#[test]
fn a_match_missing_variants_names_them() {
    let e = errors("enum E { A, B(i32), C }\nfn main() -> i32 { match B(1) { A => 0, B(x) => x } }");
    assert!(e.iter().any(|d| d.contains("`match` doesn't cover `C`")), "{e:?}");
}

#[test]
fn malformed_patterns_are_errors() {
    let e = errors("enum E { A, B(i32), C }\nfn main() -> i32 { match B(1) { A => 0, B(x, y) => x, D => 1, C => 2 } }");
    assert!(e.iter().any(|d| d.contains("`B` holds 1 value(s), the pattern binds 2")), "{e:?}");
    assert!(e.iter().any(|d| d.contains("no variant `D`")), "{e:?}");
    let e = errors("enum E { A, B(i32) }\nfn main() -> i32 { match A { _ => 1, A => 0 } }");
    assert!(e.iter().any(|d| d.contains("never reached")), "{e:?}");
    let e = errors("enum E { A, B(i32) }\nfn main() -> i32 { match A { A => 1, A => 0, B(x) => x } }");
    assert!(e.iter().any(|d| d.contains("`A` is matched twice")), "{e:?}");
    let e = errors("enum E { A }\nenum F { B }\nfn main() -> i32 { match A { A => 1, B => 0 } }");
    assert!(e.iter().any(|d| d.contains("`B` is a variant of `F`, not of `E`")), "{e:?}");
}

#[test]
fn two_enums_cannot_share_a_variant_name() {
    let e = errors("enum E { A, B(i32) }\nenum F { B }\nfn main() -> i32 { 0 }");
    assert!(e.iter().any(|d| d.contains("variant `B` is declared by `E` already")), "{e:?}");
}
