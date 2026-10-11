//! A kernel compiled to SPIR-V by MLIR, dispatched on every Vulkan device of
//! the machine, the validation layer on.

use cleave_mlir::Context;
use cleave_mlir::dialect::DialectRegistry;
use cleave_mlir::ir::Module;
use cleave_mlir::utility::register_all_dialects;
use cleave_vk::{Gpu, Options};

fn add_kernel_spirv() -> Vec<u32> {
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    let context = Context::new();
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
    let module = Module::parse(&context, include_str!("../../cleave-mlir/tests/data/add_f32.mlir"))
        .expect("failed to parse the kernel");
    // SAFETY: `module` is a valid module, owned here.
    unsafe { cleave_mlir::serialize_spirv(module.to_raw(), "") }.expect("serialization failed")
}

/// `c = a + b` over 1024 floats (16 workgroups of 64) computes the sum on
/// each device, two vendors on the development machine, without a
/// validation message. Skipped, saying so, on a machine without Vulkan.
#[test]
fn an_elementwise_kernel_computes_the_sum_on_every_device() {
    let devices = match cleave_vk::devices() {
        Ok(devices) if !devices.is_empty() => devices,
        other => {
            eprintln!("skipped: no Vulkan device ({other:?})");
            return;
        }
    };
    let spirv = add_kernel_spirv();
    const N: usize = 1024;
    let a: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..N).map(|i| 0.5 * i as f32 + 1.0).collect();
    for index in 0..devices.len() {
        let gpu = Gpu::open(&Options { device: Some(index.to_string()), validation: true })
            .unwrap_or_else(|e| panic!("opening device {index}: {e}"));
        let name = gpu.info().name.clone();
        let kernel = gpu.kernel(&spirv, "add", 3).unwrap_or_else(|e| panic!("{name}: {e}"));
        let mut buffer_a = gpu.buffer(N * 4).unwrap();
        let mut buffer_b = gpu.buffer(N * 4).unwrap();
        let buffer_c = gpu.buffer(N * 4).unwrap();
        buffer_a.write(&a);
        buffer_b.write(&b);
        gpu.dispatch(&kernel, &[&buffer_a, &buffer_b, &buffer_c], [(N / 64) as u32, 1, 1])
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let mut c = vec![0f32; N];
        buffer_c.read(&mut c);
        for i in 0..N {
            assert_eq!(c[i], a[i] + b[i], "{name}: c[{i}]");
        }
        drop((kernel, buffer_a, buffer_b, buffer_c));
        let messages = gpu.validation_messages();
        assert!(messages.is_empty(), "{name}: validation messages:\n{}", messages.join("\n"));
        eprintln!("ok on {name}");
    }
}
