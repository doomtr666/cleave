use cleave::ast::{ItemKind, Program};
use cleave::diag::Diagnostic;
use cleave::driver::compile;
use cleave::registry::Registry;
use std::path::PathBuf;

fn compile_ok(entries: Vec<(String, String)>, search_paths: &[PathBuf]) -> Program {
    let (result, _sources) = compile(entries, search_paths);
    result.unwrap_or_else(|e| panic!("{e:?}"))
}

fn compile_err(entries: Vec<(String, String)>, search_paths: &[PathBuf]) -> Vec<Diagnostic> {
    let (result, _sources) = compile(entries, search_paths);
    result.unwrap_err()
}

/// A throwaway project directory under the OS temp dir, holding one or more
/// synthetic crate subdirectories — cleaned up on drop so tests don't leave
/// litter behind regardless of pass/fail.
struct TempProject {
    root: PathBuf,
}

impl TempProject {
    fn new(name: &str) -> Self {
        let root = std::env::temp_dir().join(format!("cleave_test_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        TempProject { root }
    }

    fn write_crate_file(&self, crate_name: &str, file_name: &str, contents: &str) {
        let dir = self.root.join(crate_name);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(file_name), contents).unwrap();
    }

    fn search_paths(&self) -> Vec<PathBuf> {
        vec![self.root.clone()]
    }
}

impl Drop for TempProject {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

#[test]
fn num_is_available_without_any_explicit_use() {
    // `num` is in the prelude (`driver::PRELUDE_CRATES`) — a program that
    // never writes `use num;` at all must still see its `Num` algebra,
    // same as Rust's own prelude never needing `use std::vec::Vec;`.
    let program = compile_ok(vec![("main.cleave".to_string(), "fn f() -> i32 { 0 }".to_string())], &[]);
    let registry = Registry::build(&program);
    assert!(registry.has_algebra("Num"), "the prelude should have loaded `num` without an explicit `use`");
}

#[test]
fn use_num_pulls_in_the_real_stdlib() {
    let src = "use num;\nfn f() -> i32 { 0 }".to_string();
    let program = compile_ok(vec![("main.cleave".to_string(), src)], &[]);
    let registry = Registry::build(&program);
    assert!(registry.has_algebra("Num"), "`use num;` should pull in stdlib's Num algebra");
}

#[test]
fn qualified_use_path_behaves_identically_to_bare_crate_name() {
    let project = TempProject::new("qualified_use");
    project.write_crate_file("linalg", "linalg.cleave", "algebra VectorSpace<T> {}");

    let src = "use linalg::Matrix;\nfn f() -> i32 { 0 }".to_string();
    let program = compile_ok(vec![("main.cleave".to_string(), src)], &project.search_paths());
    let registry = Registry::build(&program);
    assert!(registry.has_algebra("VectorSpace"), "only the first path segment should matter for resolution");
}

#[test]
fn unknown_crate_reports_a_located_diagnostic() {
    let src = "use nonexistent_crate_xyz;\nfn f() -> i32 { 0 }".to_string();
    let errs = compile_err(vec![("main.cleave".to_string(), src)], &[]);
    assert_eq!(errs.len(), 1);
    assert!(errs[0].message.contains("nonexistent_crate_xyz"), "got: {}", errs[0].message);
}

#[test]
fn source_map_is_usable_even_when_compile_fails() {
    // `compile` used to only hand back a `SourceMap` on success — a CLI
    // that failed to compile could never render its own errors with
    // `file:line:col` locations. Confirms the entry file is still
    // registered in the map on the error path.
    let src = "use nonexistent_crate_xyz;\nfn f() -> i32 { 0 }".to_string();
    let (result, sources) = compile(vec![("main.cleave".to_string(), src)], &[]);
    let errs = result.unwrap_err();
    let rendered = sources.render(&errs[0]);
    assert!(rendered.starts_with("main.cleave:1:"), "got: {rendered}");
}

#[test]
fn a_crates_own_files_are_merged_before_joining_the_program() {
    let project = TempProject::new("multi_file_crate");
    project.write_crate_file("shapes", "sig.cleave", "algebra Shapes<T> { fn add(a: T, b: T) -> T; }");
    project.write_crate_file(
        "shapes",
        "axiom.cleave",
        "algebra Shapes<T> { axiom comm(a: T, b: T): add(a,b) == add(b,a); }",
    );

    let src = "use shapes;\nfn f() -> i32 { 0 }".to_string();
    let program = compile_ok(vec![("main.cleave".to_string(), src)], &project.search_paths());

    // Filtered by name, not just `ItemKind::Algebra` — `num` (with its own
    // `Num` algebra) is always loaded too now, via the prelude.
    let shapes_items: Vec<_> = program
        .items
        .iter()
        .filter(|i| matches!(&i.kind, ItemKind::Algebra(d) if d.name == "Shapes"))
        .collect();
    assert_eq!(shapes_items.len(), 1, "the two fragments must merge into one Shapes, not stay separate");
}

#[test]
fn the_same_crate_used_twice_is_only_loaded_once() {
    let project = TempProject::new("dedup_use");
    project.write_crate_file("shapes", "shapes.cleave", "algebra Shapes<T> { fn add(a: T, b: T) -> T; }");

    let a = "use shapes;\nfn f() -> i32 { 0 }".to_string();
    let b = "use shapes;\nfn g() -> i32 { 0 }".to_string();
    let program =
        compile_ok(vec![("a.cleave".to_string(), a), ("b.cleave".to_string(), b)], &project.search_paths());

    // Filtered by name, not just `ItemKind::Algebra` — `num` (with its own
    // `Num` algebra) is always loaded too now, via the prelude.
    let shapes_items: Vec<_> = program
        .items
        .iter()
        .filter(|i| matches!(&i.kind, ItemKind::Algebra(d) if d.name == "Shapes"))
        .collect();
    assert_eq!(shapes_items.len(), 1, "loading `shapes` twice must not duplicate its declarations");
}

#[test]
fn node_ids_stay_unique_across_multiple_files() {
    // Regression test: `NodeId`s used to restart at 0 in every file's own
    // `Lowerer` — harmless as long as nothing ever built a NodeId-keyed
    // side-table spanning more than one file, which is exactly what
    // `infer.rs`'s `node_types` now does. `compile` threads one shared
    // `NodeIdGen` across the entry file *and* every crate file it loads.
    let project = TempProject::new("node_id_uniqueness");
    project.write_crate_file("shapes", "shapes.cleave", "algebra Shapes<T> { fn add(a: T, b: T) -> T; }");

    let src = "use shapes;\nfn f() -> i32 { 0 }\nfn g() -> i32 { 1 }".to_string();
    let program = compile_ok(vec![("main.cleave".to_string(), src)], &project.search_paths());

    let ids: Vec<u32> = program.items.iter().map(|i| i.id.0).collect();
    let mut sorted = ids.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted.len(), ids.len(), "every item across entry + used crate must have a distinct NodeId, got {ids:?}");
}

#[test]
fn project_search_path_is_tried_before_stdlib() {
    // A project-local crate happens to share a name with a stdlib one —
    // the project path (passed first) must win, not the stdlib fallback.
    let project = TempProject::new("shadow_stdlib");
    project.write_crate_file("num", "num.cleave", "algebra Shadowed {}");

    let src = "use num;\nfn f() -> i32 { 0 }".to_string();
    let program = compile_ok(vec![("main.cleave".to_string(), src)], &project.search_paths());
    let registry = Registry::build(&program);
    assert!(registry.has_algebra("Shadowed"), "project search path should win over the stdlib fallback");
    assert!(!registry.has_algebra("Num"), "the real stdlib `num` should not have been loaded instead");
}
