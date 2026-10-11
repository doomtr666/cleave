//! SPIR-V serialization (`serialize_spirv`), on a hand-written kernel.

use cleave_mlir::Context;
use cleave_mlir::dialect::DialectRegistry;
use cleave_mlir::ir::Module;
use cleave_mlir::utility::register_all_dialects;

fn context() -> Context {
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    let context = Context::new();
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
    context
}

/// A `spirv.module` serializes to a SPIR-V binary: the magic number first,
/// the entry point's name in it.
#[test]
fn a_spirv_module_serializes_to_a_spirv_binary() {
    let context = context();
    let module = Module::parse(&context, include_str!("data/add_f32.mlir")).expect("failed to parse the kernel");
    // SAFETY: `module` is a valid module, owned here.
    let words = unsafe { cleave_mlir::serialize_spirv(module.to_raw(), "") }.expect("serialization failed");
    assert_eq!(words.first(), Some(&0x0723_0203), "not a SPIR-V binary");
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    assert!(bytes.windows(4).any(|w| w == b"add\0"), "the entry point `add` isn't in the binary");
}

/// Naming a `spirv.module` that isn't there is an error, not a crash.
#[test]
fn serializing_a_missing_spirv_module_is_an_error() {
    let context = context();
    let module = Module::parse(&context, include_str!("data/add_f32.mlir")).expect("failed to parse the kernel");
    // SAFETY: `module` is a valid module, owned here.
    let error = unsafe { cleave_mlir::serialize_spirv(module.to_raw(), "nowhere") }.unwrap_err();
    assert!(error.contains("no spirv.module named nowhere"), "{error}");
}
