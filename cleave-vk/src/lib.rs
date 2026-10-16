//! cleave's Vulkan runtime (`doc/plan-vulkan.md`, Phase 0): one device, its
//! memory through VMA, compute kernels from SPIR-V, and a synchronous
//! dispatch. Asynchrony (command buffers per sequence, timeline semaphores,
//! deferred frees) is Phase 2.
//!
//! The device is chosen explicitly, never by enumeration order: the one
//! `CLEAVE_VK_DEVICE` names (an index into [`devices`], or part of a device
//! name), else the first discrete GPU, else the first device.
//! `CLEAVE_VK_VALIDATION=1` turns the Khronos validation layer on; its
//! messages are kept and can be read with [`Gpu::validation_messages`].

use ash::vk;
use std::ffi::{CStr, CString, c_void};
use std::sync::Mutex;
use vk_mem::Alloc;

/// An error from the runtime: what failed, with Vulkan's result code when
/// there is one.
#[derive(Debug)]
pub struct Error(pub String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

fn fail(what: &str) -> impl FnOnce(vk::Result) -> Error + '_ {
    move |result| Error(format!("{what}: {result}"))
}

/// A Vulkan device as the loader reports it.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub name: String,
    pub discrete: bool,
    /// The cooperative-matrix products the device runs (`VK_KHR_cooperative_matrix`), empty
    /// without the extension: what a tensor-core kernel can be built from
    /// (`doc/plan-quantization.md`, phase 0).
    pub cooperative_matrices: Vec<CooperativeMatrixShape>,
    /// Whether kernels can hold bfloat16 values (`VK_KHR_shader_bfloat16`'s `shaderBFloat16Type`):
    /// what a program with `bf16` tensors needs on this device.
    pub bf16: bool,
}

/// One cooperative-matrix product a device supports: `result[M, N] = A[M, K] B[K, N] + C[M, N]`
/// with these element types, over a subgroup (`scope`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CooperativeMatrixShape {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub a: String,
    pub b: String,
    pub c: String,
    pub result: String,
    pub saturating: bool,
    pub scope: String,
}

impl std::fmt::Display for CooperativeMatrixShape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}x{}x{} {}*{}+{} -> {} ({}{})",
            self.m,
            self.n,
            self.k,
            self.a,
            self.b,
            self.c,
            self.result,
            self.scope,
            if self.saturating { ", saturating" } else { "" }
        )
    }
}

/// `VK_KHR_shader_bfloat16`, newer than `ash` 0.38: its name and its features structure
/// (`VkPhysicalDeviceShaderBfloat16FeaturesKHR`, structure type 1000141000 in the Vulkan registry).
const SHADER_BFLOAT16_NAME: &CStr = c"VK_KHR_shader_bfloat16";

#[repr(C)]
#[derive(Clone, Copy)]
struct ShaderBfloat16Features {
    s_type: vk::StructureType,
    p_next: *mut std::ffi::c_void,
    shader_bfloat16_type: vk::Bool32,
    shader_bfloat16_dot_product: vk::Bool32,
    shader_bfloat16_cooperative_matrix: vk::Bool32,
}

impl Default for ShaderBfloat16Features {
    fn default() -> Self {
        Self {
            s_type: vk::StructureType::from_raw(1000141000),
            p_next: std::ptr::null_mut(),
            shader_bfloat16_type: 0,
            shader_bfloat16_dot_product: 0,
            shader_bfloat16_cooperative_matrix: 0,
        }
    }
}

/// `VK_NV_cooperative_matrix2`, newer than `ash` 0.38: workgroup-scope cooperative matrices and
/// tensor addressing (`VkPhysicalDeviceCooperativeMatrix2FeaturesNV`, structure type 1000593000
/// in the Vulkan registry). Enabled for measurement (`doc/plan-quantization.md`, phase 0).
const COOPERATIVE_MATRIX2_NAME: &CStr = c"VK_NV_cooperative_matrix2";

#[repr(C)]
#[derive(Clone, Copy)]
struct CooperativeMatrix2Features {
    s_type: vk::StructureType,
    p_next: *mut std::ffi::c_void,
    workgroup_scope: vk::Bool32,
    flexible_dimensions: vk::Bool32,
    reductions: vk::Bool32,
    conversions: vk::Bool32,
    per_element_operations: vk::Bool32,
    tensor_addressing: vk::Bool32,
    block_loads: vk::Bool32,
}

impl Default for CooperativeMatrix2Features {
    fn default() -> Self {
        Self {
            s_type: vk::StructureType::from_raw(1000593000),
            p_next: std::ptr::null_mut(),
            workgroup_scope: 0,
            flexible_dimensions: 0,
            reductions: 0,
            conversions: 0,
            per_element_operations: 0,
            tensor_addressing: 0,
            block_loads: 0,
        }
    }
}

/// The optional device features `Gpu::new` enables where they exist: queried from the device,
/// then the same values handed to `vkCreateDevice` (only these fields, every other feature off).
#[derive(Clone, Default)]
struct OptionalFeatures {
    v11: vk::PhysicalDeviceVulkan11Features<'static>,
    v12: vk::PhysicalDeviceVulkan12Features<'static>,
    coopmat: vk::PhysicalDeviceCooperativeMatrixFeaturesKHR<'static>,
    bf16: ShaderBfloat16Features,
    statistics: vk::PhysicalDevicePipelineExecutablePropertiesFeaturesKHR<'static>,
    coopmat2: CooperativeMatrix2Features,
}

impl OptionalFeatures {
    fn query(
        instance: &ash::Instance,
        physical: vk::PhysicalDevice,
        coopmat: bool,
        bf16: bool,
        statistics: bool,
        coopmat2: bool,
    ) -> Self {
        let mut all = Self::default();
        let mut features2 = vk::PhysicalDeviceFeatures2::default();
        all.link(&mut features2, coopmat, bf16, statistics, coopmat2);
        if unsafe { instance.get_physical_device_properties(physical) }.api_version >= vk::API_VERSION_1_2 {
            unsafe { instance.get_physical_device_features2(physical, &mut features2) };
        }
        // Only what kernels may need; the rest stays off.
        let mut wanted = Self::default();
        wanted.v11.storage_buffer16_bit_access = all.v11.storage_buffer16_bit_access;
        wanted.v12.vulkan_memory_model = all.v12.vulkan_memory_model;
        wanted.v12.vulkan_memory_model_device_scope = all.v12.vulkan_memory_model_device_scope;
        wanted.v12.shader_float16 = all.v12.shader_float16;
        wanted.coopmat.cooperative_matrix = all.coopmat.cooperative_matrix;
        wanted.bf16.shader_bfloat16_type = all.bf16.shader_bfloat16_type;
        wanted.bf16.shader_bfloat16_cooperative_matrix = all.bf16.shader_bfloat16_cooperative_matrix;
        wanted.statistics.pipeline_executable_info = all.statistics.pipeline_executable_info;
        wanted.coopmat2 = CooperativeMatrix2Features { p_next: std::ptr::null_mut(), ..all.coopmat2 };
        wanted
    }

    /// Chains the structures behind `head`: Vulkan 1.1, 1.2, then the extensions present.
    fn link(
        &mut self,
        head: &mut vk::PhysicalDeviceFeatures2,
        coopmat: bool,
        bf16: bool,
        statistics: bool,
        coopmat2: bool,
    ) {
        let mut next: *mut std::ffi::c_void = std::ptr::null_mut();
        if coopmat2 {
            self.coopmat2.p_next = next;
            next = (&raw mut self.coopmat2).cast();
        }
        if statistics {
            self.statistics.p_next = next;
            next = (&raw mut self.statistics).cast();
        }
        if bf16 {
            self.bf16.p_next = next;
            next = (&raw mut self.bf16).cast();
        }
        if coopmat {
            self.coopmat.p_next = next;
            next = (&raw mut self.coopmat).cast();
        }
        self.v12.p_next = next;
        self.v11.p_next = (&raw mut self.v12).cast();
        head.p_next = (&raw mut self.v11).cast();
    }
}

/// A cooperative-matrix element type's name. `ash` 0.38 (Vulkan 1.3.281) predates bfloat16
/// (`VK_KHR_shader_bfloat16`) and fp8 (`VK_EXT_shader_float8`), whose values come from the
/// Vulkan registry; anything else unknown is shown by its raw value.
fn component_type_name(t: vk::ComponentTypeKHR) -> String {
    match t.as_raw() {
        0 => "f16".into(),
        1 => "f32".into(),
        2 => "f64".into(),
        3 => "i8".into(),
        4 => "i16".into(),
        5 => "i32".into(),
        6 => "i64".into(),
        7 => "u8".into(),
        8 => "u16".into(),
        9 => "u32".into(),
        10 => "u64".into(),
        1000141000 => "bf16".into(),
        1000491002 => "f8e4m3".into(),
        1000491003 => "f8e5m2".into(),
        raw => format!("type#{raw}"),
    }
}

fn cooperative_matrices(entry: &ash::Entry, instance: &ash::Instance, physical: vk::PhysicalDevice) -> Vec<CooperativeMatrixShape> {
    let supported = unsafe { instance.enumerate_device_extension_properties(physical) }.unwrap_or_default();
    let has = supported
        .iter()
        .any(|e| e.extension_name_as_c_str().is_ok_and(|n| n == ash::khr::cooperative_matrix::NAME));
    if !has {
        return Vec::new();
    }
    let loader = ash::khr::cooperative_matrix::Instance::new(entry, instance);
    let properties = unsafe { loader.get_physical_device_cooperative_matrix_properties(physical) }.unwrap_or_default();
    properties
        .iter()
        .map(|p| CooperativeMatrixShape {
            m: p.m_size,
            n: p.n_size,
            k: p.k_size,
            a: component_type_name(p.a_type),
            b: component_type_name(p.b_type),
            c: component_type_name(p.c_type),
            result: component_type_name(p.result_type),
            saturating: p.saturating_accumulation != 0,
            scope: match p.scope {
                vk::ScopeKHR::SUBGROUP => "subgroup".into(),
                vk::ScopeKHR::WORKGROUP => "workgroup".into(),
                vk::ScopeKHR::DEVICE => "device".into(),
                vk::ScopeKHR::QUEUE_FAMILY => "queue family".into(),
                other => format!("scope#{}", other.as_raw()),
            },
        })
        .collect()
}

/// Every Vulkan device the loader finds, in its order.
pub fn devices() -> Result<Vec<DeviceInfo>, Error> {
    let (entry, instance) = create_instance(false)?;
    let found = unsafe { instance.enumerate_physical_devices() }
        .map_err(fail("enumerating devices"))
        .map(|physicals| physicals.iter().map(|&p| device_info(&entry, &instance, p)).collect());
    unsafe { instance.destroy_instance(None) };
    drop(entry);
    found
}

fn device_info(entry: &ash::Entry, instance: &ash::Instance, physical: vk::PhysicalDevice) -> DeviceInfo {
    let properties = unsafe { instance.get_physical_device_properties(physical) };
    let name = properties.device_name_as_c_str().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let extensions = unsafe { instance.enumerate_device_extension_properties(physical) }.unwrap_or_default();
    let has_bf16 = extensions.iter().any(|e| e.extension_name_as_c_str().is_ok_and(|n| n == SHADER_BFLOAT16_NAME))
        && properties.api_version >= vk::API_VERSION_1_2;
    let bf16 = has_bf16 && OptionalFeatures::query(instance, physical, false, true, false, false).bf16.shader_bfloat16_type != 0;
    DeviceInfo {
        name,
        discrete: properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU,
        cooperative_matrices: cooperative_matrices(entry, instance, physical),
        bf16,
    }
}

const VALIDATION_LAYER: &CStr = c"VK_LAYER_KHRONOS_validation";

fn create_instance(validation: bool) -> Result<(ash::Entry, ash::Instance), Error> {
    // SAFETY: loading the system's Vulkan loader; nothing else runs it.
    let entry = unsafe { ash::Entry::load() }.map_err(|e| Error(format!("no Vulkan loader: {e}")))?;
    let app = vk::ApplicationInfo::default().application_name(c"cleave").api_version(vk::API_VERSION_1_3);
    let layers = if validation { vec![VALIDATION_LAYER.as_ptr()] } else { Vec::new() };
    let extensions = if validation { vec![ash::ext::debug_utils::NAME.as_ptr()] } else { Vec::new() };
    let info = vk::InstanceCreateInfo::default()
        .application_info(&app)
        .enabled_layer_names(&layers)
        .enabled_extension_names(&extensions);
    let instance = unsafe { entry.create_instance(&info, None) }.map_err(fail("creating the Vulkan instance"))?;
    Ok((entry, instance))
}

/// Validation messages, kept by the debug callback for [`Gpu::validation_messages`].
static VALIDATION_MESSAGES: Mutex<Vec<String>> = Mutex::new(Vec::new());

unsafe extern "system" fn on_validation_message(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _types: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user: *mut c_void,
) -> vk::Bool32 {
    let message = unsafe { (*data).message_as_c_str() }.map(|m| m.to_string_lossy().into_owned()).unwrap_or_default();
    if let Ok(mut messages) = VALIDATION_MESSAGES.lock() {
        messages.push(format!("{severity:?}: {message}"));
    }
    vk::FALSE
}

/// A Vulkan device ready for compute: its queue, its memory allocator, the
/// command pool its dispatches are recorded from.
pub struct Gpu {
    info: DeviceInfo,
    // Dropped explicitly, in order (`Drop`): the allocator before the device,
    // the device before the instance.
    allocator: std::mem::ManuallyDrop<vk_mem::Allocator>,
    device: ash::Device,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    /// `VK_KHR_push_descriptor`, when the device has it: a dispatch's buffers
    /// recorded into its command buffer, without a descriptor pool and set
    /// per dispatch (10 us of each of nanoLM's 400k at the full size). Up to
    /// `max_push_descriptors` bindings; a kernel with more takes a set.
    push_descriptor: Option<ash::khr::push_descriptor::Device>,
    max_push_descriptors: u32,
    /// Nanoseconds per timestamp tick (`CLEAVE_VK_PROFILE`).
    timestamp_period: f32,
    /// Whether VMA reads the system's budget (`VK_EXT_memory_budget`).
    has_budget: bool,
    /// `VK_KHR_pipeline_executable_properties`, when the device has it: every pipeline is created
    /// with its statistics captured (registers, local and shared memory... as the driver reports
    /// them), read back by [`Gpu::kernel_statistics`].
    executable_properties: Option<ash::khr::pipeline_executable_properties::Device>,
    debug: Option<(ash::ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
    instance: ash::Instance,
    _entry: ash::Entry,
}

/// How a [`Gpu`] is opened.
#[derive(Debug, Clone, Default)]
pub struct Options {
    /// An index into [`devices`], or part of a device name; `None` picks the
    /// first discrete GPU, else the first device.
    pub device: Option<String>,
    /// The Khronos validation layer, its messages kept
    /// ([`Gpu::validation_messages`]).
    pub validation: bool,
}

impl Options {
    /// `CLEAVE_VK_DEVICE` and `CLEAVE_VK_VALIDATION=1`.
    pub fn from_env() -> Options {
        Options {
            device: std::env::var("CLEAVE_VK_DEVICE").ok(),
            validation: std::env::var("CLEAVE_VK_VALIDATION").is_ok_and(|v| v == "1"),
        }
    }
}

impl Gpu {
    /// The device the environment chooses ([`Options::from_env`]).
    pub fn new() -> Result<Gpu, Error> {
        Gpu::open(&Options::from_env())
    }

    pub fn open(options: &Options) -> Result<Gpu, Error> {
        let validation = options.validation;
        let (entry, instance) = create_instance(validation)?;
        let debug = if validation {
            let utils = ash::ext::debug_utils::Instance::new(&entry, &instance);
            let info = vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::WARNING | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
                )
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                )
                .pfn_user_callback(Some(on_validation_message));
            let messenger = unsafe { utils.create_debug_utils_messenger(&info, None) }
                .map_err(fail("creating the validation messenger"))?;
            Some((utils, messenger))
        } else {
            None
        };

        let physicals = unsafe { instance.enumerate_physical_devices() }.map_err(fail("enumerating devices"))?;
        let infos: Vec<DeviceInfo> = physicals.iter().map(|&p| device_info(&entry, &instance, p)).collect();
        let chosen = match &options.device {
            Some(wanted) => match wanted.parse::<usize>() {
                Ok(index) => (index < physicals.len()).then_some(index),
                Err(_) => infos.iter().position(|d| d.name.to_lowercase().contains(&wanted.to_lowercase())),
            }
            .ok_or_else(|| Error(format!("device {wanted:?} names none of {infos:?}")))?,
            None => infos.iter().position(|d| d.discrete).unwrap_or(0),
        };
        let physical = *physicals.get(chosen).ok_or_else(|| Error("no Vulkan device".to_string()))?;

        let families = unsafe { instance.get_physical_device_queue_family_properties(physical) };
        let family = families
            .iter()
            .position(|f| f.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .ok_or_else(|| Error(format!("{} has no compute queue", infos[chosen].name)))? as u32;
        let priorities = [1.0];
        let queues = [vk::DeviceQueueCreateInfo::default().queue_family_index(family).queue_priorities(&priorities)];
        let available = unsafe { instance.enumerate_device_extension_properties(physical) }
            .map_err(fail("listing the device's extensions"))?;
        let has_push = available
            .iter()
            .any(|e| e.extension_name_as_c_str().is_ok_and(|name| name == ash::khr::push_descriptor::NAME));
        let has = |wanted: &std::ffi::CStr| {
            available.iter().any(|e| e.extension_name_as_c_str().is_ok_and(|name| name == wanted))
        };
        // `VK_EXT_memory_budget`: how much of each heap the system lets this
        // process have, for VMA's `WITHIN_BUDGET` (`create_buffer`).
        let has_budget = has(ash::ext::memory_budget::NAME);
        let mut extensions: Vec<*const std::ffi::c_char> = Vec::new();
        if has_push {
            extensions.push(ash::khr::push_descriptor::NAME.as_ptr());
        }
        if has_budget {
            extensions.push(ash::ext::memory_budget::NAME.as_ptr());
        }
        // The optional features kernels may use, each enabled where the device has it
        // (`doc/plan-quantization.md`): cooperative matrices (tensor cores), bfloat16, 16-bit
        // storage-buffer access, the Vulkan memory model the cooperative-matrix ops require.
        // Kernels that don't use them are unaffected; one that does, on a device without them,
        // is refused at pipeline creation (and by the validation layer).
        let device_version = unsafe { instance.get_physical_device_properties(physical) }.api_version;
        let has_coopmat = has(ash::khr::cooperative_matrix::NAME) && device_version >= vk::API_VERSION_1_2;
        let has_bf16 = has_coopmat && has(SHADER_BFLOAT16_NAME);
        if has_coopmat {
            extensions.push(ash::khr::cooperative_matrix::NAME.as_ptr());
        }
        if has_bf16 {
            extensions.push(SHADER_BFLOAT16_NAME.as_ptr());
        }
        let has_statistics =
            has(ash::khr::pipeline_executable_properties::NAME) && device_version >= vk::API_VERSION_1_2;
        if has_statistics {
            extensions.push(ash::khr::pipeline_executable_properties::NAME.as_ptr());
        }
        let has_coopmat2 = has_coopmat && has(COOPERATIVE_MATRIX2_NAME);
        if has_coopmat2 {
            extensions.push(COOPERATIVE_MATRIX2_NAME.as_ptr());
        }
        let features =
            OptionalFeatures::query(&instance, physical, has_coopmat, has_bf16, has_statistics, has_coopmat2);
        if has_coopmat2 && std::env::var_os("CLEAVE_VK_SHOW_FEATURES").is_some() {
            let f = &features.coopmat2;
            eprintln!(
                "VK_NV_cooperative_matrix2: workgroup scope {}, flexible dimensions {}, reductions {}, conversions {}, per-element {}, tensor addressing {}, block loads {}",
                f.workgroup_scope, f.flexible_dimensions, f.reductions, f.conversions, f.per_element_operations, f.tensor_addressing, f.block_loads
            );
        }
        let mut enabled = features.clone();
        let mut enabled2 = vk::PhysicalDeviceFeatures2::default();
        enabled.link(&mut enabled2, has_coopmat, has_bf16, has_statistics, has_coopmat2);
        let device_info = vk::DeviceCreateInfo::default().queue_create_infos(&queues).enabled_extension_names(&extensions);
        let device_info = if device_version >= vk::API_VERSION_1_2 {
            vk::DeviceCreateInfo { p_next: (&raw const enabled2).cast(), ..device_info }
        } else {
            device_info
        };
        let device =
            unsafe { instance.create_device(physical, &device_info, None) }.map_err(fail("creating the device"))?;
        let push_descriptor = has_push.then(|| ash::khr::push_descriptor::Device::new(&instance, &device));
        let max_push_descriptors = if has_push {
            let mut push = vk::PhysicalDevicePushDescriptorPropertiesKHR::default();
            let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut push);
            unsafe { instance.get_physical_device_properties2(physical, &mut properties) };
            push.max_push_descriptors
        } else {
            0
        };
        let queue = unsafe { device.get_device_queue(family, 0) };
        let command_pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
        }
        .map_err(fail("creating the command pool"))?;
        let mut allocator_info = vk_mem::AllocatorCreateInfo::new(&instance, &device, physical);
        // The device's own version, at most the 1.3 the instance asks for and
        // VMA knows: the budget extension needs 1.1's queries.
        let device_version = unsafe { instance.get_physical_device_properties(physical) }.api_version;
        allocator_info.vulkan_api_version = device_version.min(vk::API_VERSION_1_3);
        let has_budget = has_budget && device_version >= vk::API_VERSION_1_1;
        if has_budget {
            allocator_info.flags |= vk_mem::AllocatorCreateFlags::EXT_MEMORY_BUDGET;
        }
        let allocator =
            unsafe { vk_mem::Allocator::new(allocator_info) }.map_err(fail("creating the memory allocator"))?;

        let timestamp_period = unsafe { instance.get_physical_device_properties(physical) }.limits.timestamp_period;
        let executable_properties = (has_statistics && features.statistics.pipeline_executable_info != 0)
            .then(|| ash::khr::pipeline_executable_properties::Device::new(&instance, &device));
        Ok(Gpu {
            info: infos[chosen].clone(),
            allocator: std::mem::ManuallyDrop::new(allocator),
            device,
            queue,
            command_pool,
            push_descriptor,
            max_push_descriptors,
            timestamp_period,
            has_budget,
            executable_properties,
            debug,
            instance,
            _entry: entry,
        })
    }

    /// The device in use.
    pub fn info(&self) -> &DeviceInfo {
        &self.info
    }

    /// The validation layer's warnings and errors so far, taken (empty
    /// without `CLEAVE_VK_VALIDATION=1`).
    pub fn validation_messages(&self) -> Vec<String> {
        VALIDATION_MESSAGES.lock().map(|mut m| std::mem::take(&mut *m)).unwrap_or_default()
    }

    /// The device's own memory this process uses and the most the system lets
    /// it use (`VK_EXT_memory_budget`, through VMA): its device-local heaps'
    /// usage and budget, as of the last refresh (`refresh_budget`). None
    /// without the extension.
    pub fn device_memory(&self) -> Option<(u64, u64)> {
        self.device_memory_detail().map(|d| (d[0], d[1]))
    }

    /// `device_memory`, then VMA's own: the bytes of its blocks (device
    /// memory it took from the system) and of the allocations in them.
    fn device_memory_detail(&self) -> Option<[u64; 4]> {
        if !self.has_budget {
            return None;
        }
        // SAFETY: the allocator is alive as long as `self`.
        let properties = unsafe { self.allocator.get_memory_properties() };
        let budgets = self.allocator.get_heap_budgets().ok()?;
        let mut total = [0; 4];
        for (heap, budget) in properties.memory_heaps[..properties.memory_heap_count as usize].iter().zip(&budgets) {
            if heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL) {
                total[0] += budget.usage;
                total[1] += budget.budget;
                total[2] += budget.statistics.blockBytes;
                total[3] += budget.statistics.allocationBytes;
            }
        }
        Some(total)
    }

    /// Fetches the system's budget again (VMA otherwise keeps the one it
    /// read at start, plus its own allocations since: another program taking
    /// device memory goes unseen, and Windows then moves ours to the host's).
    fn refresh_budget(&self, frame: u32) {
        // SAFETY: the allocator is alive as long as `self`.
        unsafe { self.allocator.set_current_frame_index(frame) };
    }

    /// A storage buffer of `bytes` bytes the host can also read and write
    /// (mapped), in the memory the host reads fastest: for data that goes
    /// to the device for one dispatch and comes back.
    pub fn buffer(&self, bytes: usize) -> Result<Buffer<'_>, Error> {
        self.create_buffer(bytes, vk_mem::MemoryUsage::Auto, vk_mem::AllocationCreateFlags::HOST_ACCESS_RANDOM)
    }

    /// A storage buffer of `bytes` bytes in the device's own memory, not
    /// mapped for the host: for data that stays on the device across
    /// dispatches, reached by the host through transfers.
    pub fn device_buffer(&self, bytes: usize) -> Result<Buffer<'_>, Error> {
        self.create_buffer(bytes, vk_mem::MemoryUsage::AutoPreferDevice, vk_mem::AllocationCreateFlags::empty())
    }

    fn create_buffer(
        &self,
        bytes: usize,
        usage: vk_mem::MemoryUsage,
        access: vk_mem::AllocationCreateFlags,
    ) -> Result<Buffer<'_>, Error> {
        let info = vk::BufferCreateInfo::default()
            .size(bytes.max(4) as u64)
            .usage(
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let mapped_flag =
            if access.is_empty() { vk_mem::AllocationCreateFlags::empty() } else { vk_mem::AllocationCreateFlags::MAPPED };
        // Device-only buffers in the device's memory, never VMA's fallback to
        // the host's when it's full: the runtime evicts mirrors instead
        // (`Runtime::take_buffer`), where the fallback put them in memory
        // Windows then pages, a 42 s step and 21 GB of the host's for nanoLM.
        // Within the budget the system gives the process, too: past it, Windows
        // doesn't fail an allocation but pages device memory to the host's
        // behind the device's back (a 33 s step, waits for the paging).
        let (required_flags, budget) = if access.is_empty() {
            (vk::MemoryPropertyFlags::DEVICE_LOCAL, vk_mem::AllocationCreateFlags::WITHIN_BUDGET)
        } else {
            (vk::MemoryPropertyFlags::empty(), vk_mem::AllocationCreateFlags::empty())
        };
        let allocation_info = vk_mem::AllocationCreateInfo {
            usage,
            flags: access | mapped_flag | budget,
            required_flags,
            ..Default::default()
        };
        let (buffer, allocation) =
            unsafe { self.allocator.create_buffer(&info, &allocation_info) }.map_err(fail("allocating a buffer"))?;
        let mapped = self.allocator.get_allocation_info(&allocation).mapped_data as *mut u8;
        Ok(Buffer { gpu: self, buffer, allocation, bytes, mapped })
    }

    /// A compute kernel: the entry point `entry` of the SPIR-V module
    /// `spirv`, reading and writing `bindings` storage buffers (set 0,
    /// bindings `0..bindings`, in the order a dispatch passes them).
    pub fn kernel(&self, spirv: &[u32], entry: &str, bindings: u32) -> Result<Kernel<'_>, Error> {
        self.kernel_specialized(spirv, entry, bindings, &[])
    }

    /// [`Gpu::kernel`] with specialization constants: `(constant_id, value)` pairs, each a 32-bit
    /// value (a `uint`, or a `bool` as 0 or 1).
    pub fn kernel_specialized(
        &self,
        spirv: &[u32],
        entry: &str,
        bindings: u32,
        constants: &[(u32, u32)],
    ) -> Result<Kernel<'_>, Error> {
        let device = &self.device;
        let module = unsafe { device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(spirv), None) }
            .map_err(fail("creating the shader module"))?;
        let layout_bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..bindings)
            .map(|b| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(b)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        let pushed = self.push_descriptor.is_some() && bindings <= self.max_push_descriptors;
        let flags = if pushed {
            vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR
        } else {
            vk::DescriptorSetLayoutCreateFlags::empty()
        };
        let set_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().flags(flags).bindings(&layout_bindings),
                None,
            )
        }
        .map_err(fail("creating the descriptor set layout"))?;
        let set_layouts = [set_layout];
        let pipeline_layout =
            unsafe { device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts), None) }
                .map_err(fail("creating the pipeline layout"))?;
        let name = CString::new(entry).map_err(|_| Error(format!("entry point name {entry:?}")))?;
        let entries: Vec<vk::SpecializationMapEntry> = constants
            .iter()
            .enumerate()
            .map(|(i, &(id, _))| vk::SpecializationMapEntry::default().constant_id(id).offset(4 * i as u32).size(4))
            .collect();
        let values: Vec<u8> = constants.iter().flat_map(|&(_, v)| v.to_le_bytes()).collect();
        let specialization = vk::SpecializationInfo::default().map_entries(&entries).data(&values);
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(&name);
        let stage = if constants.is_empty() { stage } else { stage.specialization_info(&specialization) };
        let pipeline = unsafe {
            device.create_compute_pipelines(
                vk::PipelineCache::null(),
                &[vk::ComputePipelineCreateInfo::default().stage(stage).layout(pipeline_layout).flags(
                    if self.executable_properties.is_some() {
                        vk::PipelineCreateFlags::CAPTURE_STATISTICS_KHR
                    } else {
                        vk::PipelineCreateFlags::empty()
                    },
                )],
                None,
            )
        }
        .map_err(|(_, result)| fail("creating the compute pipeline")(result))?[0];
        unsafe { device.destroy_shader_module(module, None) };
        Ok(Kernel { gpu: self, bindings, pushed, set_layout, pipeline_layout, pipeline })
    }

    /// What the driver reports about `kernel`'s compiled code (`VK_KHR_pipeline_executable_properties`):
    /// per executable, its statistics as `(name, value)` (on NVIDIA: registers per thread, local
    /// memory, shared memory, ...). Empty without the extension.
    pub fn kernel_statistics(&self, kernel: &Kernel) -> Vec<(String, String)> {
        let Some(properties) = &self.executable_properties else { return Vec::new() };
        let info = vk::PipelineInfoKHR::default().pipeline(kernel.pipeline);
        let executables = unsafe { properties.get_pipeline_executable_properties(&info) }.unwrap_or_default();
        let text = |chars: &[std::ffi::c_char]| unsafe { CStr::from_ptr(chars.as_ptr()) }.to_string_lossy().into_owned();
        let mut out = Vec::new();
        for index in 0..executables.len() as u32 {
            let executable = vk::PipelineExecutableInfoKHR::default().pipeline(kernel.pipeline).executable_index(index);
            let statistics = unsafe { properties.get_pipeline_executable_statistics(&executable) }.unwrap_or_default();
            for s in &statistics {
                let value = unsafe {
                    match s.format {
                        vk::PipelineExecutableStatisticFormatKHR::BOOL32 => (s.value.b32 != 0).to_string(),
                        vk::PipelineExecutableStatisticFormatKHR::INT64 => s.value.i64.to_string(),
                        vk::PipelineExecutableStatisticFormatKHR::UINT64 => s.value.u64.to_string(),
                        _ => s.value.f64.to_string(),
                    }
                };
                out.push((text(&s.name), value));
            }
        }
        out
    }

    /// Runs `kernel` over `groups` workgroups with `buffers` bound in order,
    /// and waits for it to finish.
    pub fn dispatch(&self, kernel: &Kernel, buffers: &[&Buffer], groups: [u32; 3]) -> Result<(), Error> {
        if buffers.len() != kernel.bindings as usize {
            return Err(Error(format!("the kernel binds {} buffers, {} given", kernel.bindings, buffers.len())));
        }
        let device = &self.device;
        let sizes = [vk::DescriptorPoolSize::default().ty(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(kernel.bindings)];
        let pool = unsafe { device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&sizes), None) }
            .map_err(fail("creating a descriptor pool"))?;
        let result = (|| {
            let infos: Vec<[vk::DescriptorBufferInfo; 1]> = buffers
                .iter()
                .map(|b| [vk::DescriptorBufferInfo::default().buffer(b.buffer).offset(0).range(vk::WHOLE_SIZE)])
                .collect();
            let mut writes: Vec<vk::WriteDescriptorSet> = infos
                .iter()
                .enumerate()
                .map(|(i, info)| {
                    vk::WriteDescriptorSet::default()
                        .dst_binding(i as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(info)
                })
                .collect();
            // A set, unless the kernel's buffers are pushed (`Kernel::pushed`).
            let set = if kernel.pushed {
                None
            } else {
                let set_layouts = [kernel.set_layout];
                let set = unsafe {
                    device.allocate_descriptor_sets(
                        &vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&set_layouts),
                    )
                }
                .map_err(fail("allocating a descriptor set"))?[0];
                for write in &mut writes {
                    write.dst_set = set;
                }
                unsafe { device.update_descriptor_sets(&writes, &[]) };
                Some(set)
            };

            let command = unsafe {
                device.allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(self.command_pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
            }
            .map_err(fail("allocating a command buffer"))?[0];
            let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }.map_err(fail("creating a fence"))?;
            let run = (|| unsafe {
                device
                    .begin_command_buffer(command, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))
                    .map_err(fail("beginning the command buffer"))?;
                device.cmd_bind_pipeline(command, vk::PipelineBindPoint::COMPUTE, kernel.pipeline);
                match (set, &self.push_descriptor) {
                    (Some(set), _) => device.cmd_bind_descriptor_sets(
                        command,
                        vk::PipelineBindPoint::COMPUTE,
                        kernel.pipeline_layout,
                        0,
                        &[set],
                        &[],
                    ),
                    (None, Some(push)) => push.cmd_push_descriptor_set(
                        command,
                        vk::PipelineBindPoint::COMPUTE,
                        kernel.pipeline_layout,
                        0,
                        &writes,
                    ),
                    (None, None) => unreachable!("a kernel pushes its buffers only with VK_KHR_push_descriptor"),
                }
                device.cmd_dispatch(command, groups[0], groups[1], groups[2]);
                // The kernel's writes made visible to the host's reads of the
                // mapped buffers once the fence signals.
                let barrier = vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::HOST_READ);
                device.cmd_pipeline_barrier(
                    command,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::HOST,
                    vk::DependencyFlags::empty(),
                    &[barrier],
                    &[],
                    &[],
                );
                device.end_command_buffer(command).map_err(fail("ending the command buffer"))?;
                let commands = [command];
                device
                    .queue_submit(self.queue, &[vk::SubmitInfo::default().command_buffers(&commands)], fence)
                    .map_err(fail("submitting the dispatch"))?;
                device.wait_for_fences(&[fence], true, u64::MAX).map_err(fail("waiting for the dispatch"))
            })();
            unsafe {
                device.destroy_fence(fence, None);
                device.free_command_buffers(self.command_pool, &[command]);
            }
            run
        })();
        unsafe { device.destroy_descriptor_pool(pool, None) };
        result
    }

    /// Copies the first `bytes` bytes of `from` into `to` on the device, and waits: how data
    /// reaches a [`Gpu::device_buffer`] (from a mapped [`Gpu::buffer`]) and comes back.
    pub fn copy(&self, from: &Buffer, to: &Buffer, bytes: usize) -> Result<(), Error> {
        assert!(bytes <= from.bytes && bytes <= to.bytes, "{bytes} bytes copied between buffers of {} and {}", from.bytes, to.bytes);
        let device = &self.device;
        let command = unsafe {
            device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(self.command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        }
        .map_err(fail("allocating a command buffer"))?[0];
        let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }.map_err(fail("creating a fence"))?;
        let run = (|| unsafe {
            device
                .begin_command_buffer(command, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))
                .map_err(fail("beginning the command buffer"))?;
            // The host's writes to a mapped source, and earlier kernels' writes, visible to the copy;
            // the copy visible to later kernels and to the host.
            let before = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::HOST_WRITE | vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
            device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::HOST | vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[before],
                &[],
                &[],
            );
            device.cmd_copy_buffer(command, from.buffer, to.buffer, &[vk::BufferCopy::default().size(bytes as u64)]);
            let after = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::HOST_READ);
            device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[after],
                &[],
                &[],
            );
            device.end_command_buffer(command).map_err(fail("ending the command buffer"))?;
            let commands = [command];
            device
                .queue_submit(self.queue, &[vk::SubmitInfo::default().command_buffers(&commands)], fence)
                .map_err(fail("submitting the copy"))?;
            device.wait_for_fences(&[fence], true, u64::MAX).map_err(fail("waiting for the copy"))
        })();
        unsafe {
            device.destroy_fence(fence, None);
            device.free_command_buffers(self.command_pool, &[command]);
        }
        run
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_command_pool(self.command_pool, None);
            std::mem::ManuallyDrop::drop(&mut self.allocator);
            self.device.destroy_device(None);
            if let Some((utils, messenger)) = self.debug.take() {
                utils.destroy_debug_utils_messenger(messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

/// A storage buffer on a [`Gpu`], mapped for the host.
pub struct Buffer<'g> {
    gpu: &'g Gpu,
    buffer: vk::Buffer,
    allocation: vk_mem::Allocation,
    bytes: usize,
    mapped: *mut u8,
}

impl Buffer<'_> {
    /// Its size in bytes.
    pub fn len(&self) -> usize {
        self.bytes
    }

    pub fn is_empty(&self) -> bool {
        self.bytes == 0
    }

    /// Copies `data` to its start. A host-visible buffer only ([`Gpu::buffer`]),
    /// which no work in flight uses.
    pub fn write<T: Copy>(&mut self, data: &[T]) {
        let bytes = std::mem::size_of_val(data);
        assert!(bytes <= self.bytes, "{bytes} bytes written into a buffer of {}", self.bytes);
        assert!(!self.mapped.is_null(), "a device-only buffer written by the host");
        // SAFETY: `mapped` points to `self.bytes` bytes this buffer owns, which
        // no work in flight uses (the caller's guarantee).
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, self.mapped, bytes) };
        // Made visible to the device on memory that isn't coherent.
        let _ = self.gpu.allocator.flush_allocation(&self.allocation, 0, vk::WHOLE_SIZE);
    }

    /// Copies its start into `out`. A host-visible buffer only, whose
    /// writers have finished.
    pub fn read<T: Copy>(&self, out: &mut [T]) {
        let bytes = std::mem::size_of_val(out);
        assert!(bytes <= self.bytes, "{bytes} bytes read from a buffer of {}", self.bytes);
        assert!(!self.mapped.is_null(), "a device-only buffer read by the host");
        // The device's writes made visible on memory that isn't coherent.
        let _ = self.gpu.allocator.invalidate_allocation(&self.allocation, 0, vk::WHOLE_SIZE);
        // SAFETY: as in `write`.
        unsafe { std::ptr::copy_nonoverlapping(self.mapped, out.as_mut_ptr() as *mut u8, bytes) };
    }
}

// SAFETY: `mapped` is the buffer's own mapping, valid as long as the buffer;
// moving the buffer to another thread moves that ownership with it.
unsafe impl Send for Buffer<'_> {}

impl Drop for Buffer<'_> {
    fn drop(&mut self) {
        unsafe { self.gpu.allocator.destroy_buffer(self.buffer, &mut self.allocation) };
    }
}

/// A compute pipeline on a [`Gpu`].
pub struct Kernel<'g> {
    gpu: &'g Gpu,
    bindings: u32,
    /// Its buffers pushed into the command buffer (`Gpu::push_descriptor`)
    /// rather than bound as a set.
    pushed: bool,
    set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

impl Drop for Kernel<'_> {
    fn drop(&mut self) {
        let device = &self.gpu.device;
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}

/// What the runtime moved between host and device, for tests and
/// measurements ([`stats`], `CLEAVE_VK_STATS=1`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Stats {
    pub dispatches: u64,
    /// Batches of commands submitted to the device.
    pub submits: u64,
    /// Times the host waited for the device, and for how long in all.
    pub waits: u64,
    pub wait_ns: u64,
    /// Time the host spent in the runtime's calls, waits included.
    pub runtime_ns: u64,
    pub uploaded_bytes: u64,
    pub downloaded_bytes: u64,
    /// `CLEAVE_VK_PROFILE`: the time dispatches spent in each phase, waits
    /// for the device left out ([`DISPATCH_PHASES`]).
    pub dispatch_phase_ns: [u64; 6],
    /// `CLEAVE_VK_PROFILE`, waits left out: the host's synchronizations
    /// (`cleave_vk_sync_host`, `cleave_vk_host_wrote`, `cleave_vk_discard`
    /// that do something), uploads (a staging buffer made and filled) and the
    /// copies of downloads into the host's memory.
    pub sync_ns: u64,
    pub upload_ns: u64,
    pub download_copy_ns: u64,
    /// Buffers made, not taken from the spare ones.
    pub buffers_made: u64,
    /// Mirrors freed to make room in the device's memory.
    pub evictions: u64,
    /// `CLEAVE_VK_PROFILE`: what dispatches uploaded, by cause (`upload_causes`):
    /// a new mirror the kernel only writes (its contents undefined, nothing to
    /// upload: a discard the compiler missed), a new mirror it reads (the
    /// host's data), a mirror the host wrote since.
    pub upload_causes: [u64; 3],
    /// Commands recorded without a barrier before them, and with one
    /// (`Runtime::record`).
    pub barriers: [u64; 2],
    /// `CLEAVE_VK_PROFILE`: the batches' spans on the device, from a batch's
    /// first kernel's start to its last one's end, and how much of them some
    /// kernel ran (the union of their intervals).
    pub batch_span_ns: u64,
    pub batch_busy_ns: u64,
    /// The most device memory the process used (`Gpu::device_memory`), as
    /// a new buffer was made, and then: the budget, the mirrors', the spare
    /// buffers', VMA's blocks' and allocations'.
    pub device_peak: [u64; 6],
}

/// The causes [`Stats::upload_causes`] counts.
pub const UPLOAD_CAUSES: [&str; 3] = [
    "new mirrors the kernel only writes (missed discards)",
    "new mirrors the kernel reads",
    "mirrors the host wrote since",
];

/// The phases of a dispatch [`Stats::dispatch_phase_ns`] times.
pub const DISPATCH_PHASES: [&str; 6] = [
    "entry, kernel lookup",
    "mirrors (uploads, retirements)",
    "copied arguments (scalars)",
    "descriptors",
    "recording",
    "copies back, submit",
];

/// The device copy of one host payload range (`doc/plan-vulkan.md`, Phase
/// 1): the host's copy stays the one refcounted and freed; this one, in the
/// device's own memory, lets a kernel read it without an upload when the
/// host hasn't written it since (`host_dirty` false), and the host skip a
/// download it never asks for (`device_newer`).
struct Mirror {
    len: usize,
    buffer: Buffer<'static>,
    host_dirty: bool,
    device_newer: bool,
    /// When a dispatch last used it (`Runtime::uses`): the least recent go
    /// first when the device's memory runs out.
    last_use: u64,
}

/// What a batch keeps alive until the device has finished it: what its
/// commands use and nothing else holds any more.
enum Garbage {
    /// Kept spare once the device is done with it (`Runtime::spare`).
    Buffer(Buffer<'static>),
    DescriptorPool(vk::DescriptorPool),
    /// Freed once the device is done with it: a spare buffer pushed out.
    Free(Buffer<'static>),
}

/// Commands recorded into one command buffer and submitted together
/// (`doc/plan-vulkan.md`, Phase 2): dispatches, uploads, downloads, a full
/// memory barrier between any two.
struct Batch {
    command: vk::CommandBuffer,
    fence: vk::Fence,
    commands: usize,
    garbage: Vec<Garbage>,
    /// `CLEAVE_VK_PROFILE`: two timestamps around each dispatch, and which
    /// kernel each pair times.
    queries: Option<vk::QueryPool>,
    timed: Vec<usize>,
    /// The buffers commands read and wrote since the batch's last barrier
    /// (`Runtime::record`).
    read_since: std::collections::HashSet<vk::Buffer>,
    written_since: std::collections::HashSet<vk::Buffer>,
}

/// Dispatches a batch times at most (`CLEAVE_VK_PROFILE`); later ones in the
/// same batch go untimed.
const TIMED_PER_BATCH: usize = 1024;

/// What one kernel cost (`CLEAVE_VK_PROFILE`): its dispatches and the
/// device's time running them.
#[derive(Debug, Clone, Default)]
struct KernelProfile {
    name: String,
    dispatches: u64,
    timed: u64,
    gpu_ns: f64,
    /// Bytes its arguments not mirrored were copied in, and back.
    copied_in: u64,
    copied_out: u64,
    /// Bytes its mirrors were uploaded again, and downloaded to make room.
    uploaded: u64,
    downloaded: u64,
    /// Of `uploaded`, by cause (`Stats::upload_causes`).
    upload_causes: [u64; 3],
}

/// A batch is submitted, without waiting for it, once it holds this many
/// commands: the device starts on long sequences before the host needs
/// anything back, and what a batch keeps alive stays bounded.
const BATCH_COMMANDS: usize = 256;

/// Batches submitted and not finished at most (`Runtime::submit`): enough
/// for the device never to wait for the host, few enough that what they keep
/// alive stays small.
const MOST_IN_FLIGHT: usize = 4;

/// The runtime generated code calls: the device, opened on first use
/// ([`Gpu::new`]), the kernels built so far (by the address of their SPIR-V
/// binary and their entry point), the mirrors (by host address), the batch
/// being recorded and those submitted and not yet waited for.
struct Runtime {
    gpu: &'static Gpu,
    kernels: std::collections::HashMap<(usize, String), Kernel<'static>>,
    mirrors: std::collections::BTreeMap<usize, Mirror>,
    recording: Option<Batch>,
    in_flight: Vec<Batch>,
    stats: Stats,
    /// Each kernel's index into `profile`, by the same key as `kernels`.
    kernel_ids: std::collections::HashMap<(usize, String), usize>,
    profile: Vec<KernelProfile>,
    profiling: bool,
    /// Profiling: the downloads host reads caused, by the function reading
    /// (count, bytes).
    host_reads: std::collections::HashMap<String, (u64, u64)>,
    /// Buffers whose last use the device has finished, kept for the next
    /// one of the same size and kind (mapped for the host or not) instead of
    /// freed: a step makes and frees the same sizes thousands of times (each
    /// fresh tensor's mirror, each transfer's staging), each a VMA
    /// allocation. `spare_bytes` of them, device and host memory apart,
    /// up to `SPARE_DEVICE_BYTES` and `SPARE_HOST_BYTES`.
    spare: std::collections::HashMap<(usize, bool), Vec<Buffer<'static>>>,
    spare_bytes: [usize; 2],
    /// The sizes of the spare buffers in the order they were kept, device
    /// and host apart: the oldest go when one more doesn't fit (a full
    /// spare refusing new buffers kept sizes no one asked for again, while
    /// split products' partials, 8 MiB each, were made and freed 50k times).
    /// An entry whose buffer was taken since is skipped.
    spare_order: [std::collections::VecDeque<usize>; 2],
    /// A clock of mirror uses (`Mirror::last_use`).
    uses: u64,
}

/// How much memory `Runtime::spare` keeps: the device's freed first when
/// its memory runs out (`take_buffer`), so what's kept never pushes
/// mirrors out; the host's for staging only.
const SPARE_DEVICE_BYTES: usize = 1 << 30;
const SPARE_HOST_BYTES: usize = 256 << 20;

static RUNTIME: Mutex<Option<Runtime>> = Mutex::new(None);

/// Bumped whenever a mirror may newly need a download: a kernel wrote it, or
/// mirrors came or went. A host synchronization of a range already done at
/// the current value has nothing to do (`recently`).
static DEVICE_GENERATION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Bumped whenever a mirror may newly need marking stale: it was uploaded,
/// or mirrors came or went. A `cleave_vk_host_wrote` of a range already
/// marked at the current value has nothing to do.
static HOST_GENERATION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn bump(generation: &std::sync::atomic::AtomicU64) {
    generation.fetch_add(1, std::sync::atomic::Ordering::Release);
}

/// The last few ranges this thread synchronized (`SYNCED`) or marked
/// (`MARKED`), with the generation they were done at: a host loop reporting
/// the same range at each element (when the compiler couldn't hoist the
/// report out of it) costs a lookup here, not the lock.
type RecentRanges = std::cell::Cell<[(i64, i64, u64); 8]>;
thread_local! {
    static SYNCED: RecentRanges = const { std::cell::Cell::new([(0, -1, 0); 8]) };
    static MARKED: RecentRanges = const { std::cell::Cell::new([(0, -1, 0); 8]) };
}

/// Whether `[ptr, ptr + bytes)` is in `recent` at `generation`; if not, the
/// caller does the work, then `remember`s it.
fn recently(recent: &'static std::thread::LocalKey<RecentRanges>, ptr: i64, bytes: i64, generation: u64) -> bool {
    recent.with(|r| r.get().iter().any(|&entry| entry == (ptr, bytes, generation)))
}

fn remember(recent: &'static std::thread::LocalKey<RecentRanges>, ptr: i64, bytes: i64, generation: u64) {
    recent.with(|r| {
        let mut entries = r.get();
        entries.rotate_right(1);
        entries[0] = (ptr, bytes, generation);
        r.set(entries);
    });
}

/// Calls of `cleave_vk_sync_host` and `cleave_vk_host_wrote` (`CLEAVE_VK_PROFILE`).
static SYNC_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static WROTE_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Whether any mirror exists: the fast path of the free hook and of the host
/// synchronizations, so that a program whose payloads the device never saw
/// doesn't take the lock.
static MIRRORS_LIVE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn abort_with(message: String) -> ! {
    eprintln!("cleave Vulkan runtime: {message}");
    std::process::abort()
}

/// `CLEAVE_VK_PARANOID=upload|download|both`, a diagnostic: every mirror
/// uploaded before each dispatch (a host write the compiler didn't report
/// then can't leave the device stale), or downloaded after it (a host read
/// it didn't report can't see a stale host copy). A run that matches the
/// CPU's only with one of them on points at the direction of a missing
/// synchronization.
fn paranoid() -> (bool, bool) {
    static MODE: std::sync::OnceLock<(bool, bool)> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("CLEAVE_VK_PARANOID").as_deref() {
        Ok("upload") => (true, false),
        Ok("download") => (false, true),
        Ok("both") => (true, true),
        _ => (false, false),
    })
}

/// `CLEAVE_VK_CHECK_NAN=1`, a diagnostic: after each dispatch, what the
/// kernel wrote is brought back and scanned as `f32`s, the first kernel to
/// write a NaN or an infinity named (once) on stderr.
fn checking_nans() -> bool {
    static CHECK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CHECK.get_or_init(|| std::env::var("CLEAVE_VK_CHECK_NAN").is_ok_and(|v| v == "1"))
}

/// `CLEAVE_VK_DEVICE_BUDGET=<bytes>`: the device memory mirrors may hold
/// together (`Runtime::take_buffer`); unbounded by default.
fn device_budget() -> Option<usize> {
    static BUDGET: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *BUDGET.get_or_init(|| std::env::var("CLEAVE_VK_DEVICE_BUDGET").ok().and_then(|v| v.parse().ok()))
}

/// `CLEAVE_VK_FULL_BARRIERS=1`: a barrier before every command
/// (`Runtime::record`), a diagnostic for a result the skipped ones change.
fn full_barriers() -> bool {
    static FULL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FULL.get_or_init(|| std::env::var("CLEAVE_VK_FULL_BARRIERS").is_ok_and(|v| v == "1"))
}

fn tracing() -> bool {
    static TRACE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *TRACE.get_or_init(|| std::env::var("CLEAVE_VK_TRACE").is_ok_and(|v| v == "1"))
}

fn runtime(guard: &mut Option<Runtime>) -> &mut Runtime {
    guard.get_or_insert_with(|| {
        let gpu = Gpu::new().unwrap_or_else(|e| abort_with(format!("no device: {e}")));
        cleave_rt::set_free_hook(on_freed);
        Runtime {
            gpu: Box::leak(Box::new(gpu)),
            kernels: Default::default(),
            mirrors: Default::default(),
            recording: None,
            in_flight: Vec::new(),
            stats: Stats::default(),
            kernel_ids: Default::default(),
            profile: Vec::new(),
            host_reads: Default::default(),
            spare: Default::default(),
            spare_bytes: [0, 0],
            spare_order: Default::default(),
            uses: 0,
            profiling: {
                let profiling = std::env::var("CLEAVE_VK_PROFILE").is_ok_and(|v| v == "1");
                if profiling {
                    // SAFETY: `report` is a plain function, callable at exit.
                    unsafe { atexit(report) };
                }
                profiling
            },
        }
    })
}

unsafe extern "C" {
    fn atexit(f: extern "C" fn()) -> std::ffi::c_int;
}

/// `CLEAVE_VK_PROFILE=1`, at the process's exit: what moved, how often and
/// how long the host waited, and each kernel's dispatches and device time,
/// most expensive first.
extern "C" fn report() {
    let mut guard = lock();
    let Some(runtime) = guard.as_mut() else { return };
    runtime.finish();
    let s = runtime.stats;
    let mut out = String::new();
    use std::fmt::Write as _;
    let _ = writeln!(
        out,
        "cleave Vulkan profile: {} dispatches, {} submits, {} waits ({:.1} ms waiting), {:.1} ms in the runtime, {:.1} MiB up, {:.1} MiB down",
        s.dispatches,
        s.submits,
        s.waits,
        s.wait_ns as f64 / 1e6,
        s.runtime_ns as f64 / 1e6,
        s.uploaded_bytes as f64 / 1048576.0,
        s.downloaded_bytes as f64 / 1048576.0
    );
    let phases: Vec<String> = DISPATCH_PHASES
        .iter()
        .zip(s.dispatch_phase_ns)
        .map(|(name, ns)| format!("{name} {:.1} ms", ns as f64 / 1e6))
        .collect();
    let _ = writeln!(
        out,
        "  dispatches, waits left out: {:.1} us each ({})",
        s.dispatch_phase_ns.iter().sum::<u64>() as f64 / 1e3 / s.dispatches.max(1) as f64,
        phases.join(", ")
    );
    let _ = writeln!(
        out,
        "  waits left out: {:.1} ms in host synchronizations, {:.1} ms making uploads, {:.1} ms copying downloads; {} buffers made, {} mirrors evicted",
        s.sync_ns as f64 / 1e6,
        s.upload_ns as f64 / 1e6,
        s.download_copy_ns as f64 / 1e6,
        s.buffers_made,
        s.evictions
    );
    let _ = writeln!(
        out,
        "  host synchronizations: {} sync_host, {} host_wrote calls",
        SYNC_CALLS.load(std::sync::atomic::Ordering::Relaxed),
        WROTE_CALLS.load(std::sync::atomic::Ordering::Relaxed)
    );
    let mut samples: Vec<(String, u64)> =
        SYNC_SAMPLES.lock().unwrap_or_else(|p| p.into_inner()).iter().map(|(k, v)| (k.clone(), *v)).collect();
    samples.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
    for (caller, calls) in samples.iter().take(10) {
        let _ = writeln!(out, "    ~{calls:>9} {caller}");
    }
    let total: f64 = runtime.profile.iter().map(|k| k.gpu_ns).sum();
    let _ = writeln!(out, "  device time in kernels: {:.1} ms", total / 1e6);
    let _ = writeln!(
        out,
        "  barriers: {} commands after one, {} without ({:.0}%: no buffer hazard)",
        s.barriers[1],
        s.barriers[0],
        100.0 * s.barriers[0] as f64 / (s.barriers[0] + s.barriers[1]).max(1) as f64
    );
    let idle = s.batch_span_ns.saturating_sub(s.batch_busy_ns);
    let _ = writeln!(
        out,
        "  device time across batches: {:.1} ms, {:.1} ms of it running a kernel, {:.1} ms ({:.0}%) between kernels",
        s.batch_span_ns as f64 / 1e6,
        s.batch_busy_ns as f64 / 1e6,
        idle as f64 / 1e6,
        100.0 * idle as f64 / s.batch_span_ns.max(1) as f64
    );
    let [used, budget, mirrors, spare, blocks, allocations] = s.device_peak.map(|b| b as f64 / 1048576.0);
    let _ = writeln!(
        out,
        "  device memory at its peak: {used:.0} MiB of a {budget:.0} MiB budget: {mirrors:.0} MiB of mirrors, \
         {spare:.0} MiB of spare buffers, {:.0} MiB else (allocations in VMA's blocks {allocations:.0} MiB of \
         {blocks:.0} MiB); {} mirrors evicted",
        used - mirrors - spare,
        s.evictions
    );
    let mut kernels: Vec<&KernelProfile> = runtime.profile.iter().collect();
    kernels.sort_by(|a, b| b.gpu_ns.total_cmp(&a.gpu_ns));
    let mib = |b: u64| b as f64 / 1048576.0;
    let (mut uploaded, mut downloaded) = (0, 0);
    for k in &runtime.profile {
        uploaded += k.copied_in + k.uploaded;
        downloaded += k.copied_out + k.downloaded;
    }
    let _ = writeln!(
        out,
        "  transfers: {:.1} MiB up and {:.1} MiB down at dispatches, {:.1} MiB down for host reads",
        mib(uploaded),
        mib(downloaded),
        mib(s.downloaded_bytes.saturating_sub(downloaded))
    );
    let mut readers: Vec<(&String, &(u64, u64))> = runtime.host_reads.iter().collect();
    readers.sort_by_key(|(_, (_, bytes))| std::cmp::Reverse(*bytes));
    for (name, (reads, bytes)) in readers.iter().take(10) {
        let _ = writeln!(out, "  {:>10.1} MiB down in {:>8} host reads  {}", mib(*bytes), reads, name);
    }
    let causes: Vec<String> = UPLOAD_CAUSES
        .iter()
        .zip(s.upload_causes)
        .map(|(cause, bytes)| format!("{:.1} MiB {cause}", mib(bytes)))
        .collect();
    let _ = writeln!(out, "  uploads at dispatches: {}", causes.join(", "));
    for (cause, name) in UPLOAD_CAUSES.iter().enumerate() {
        let mut by: Vec<&KernelProfile> = runtime.profile.iter().filter(|k| k.upload_causes[cause] > 0).collect();
        by.sort_by_key(|k| std::cmp::Reverse(k.upload_causes[cause]));
        for k in by.iter().take(5) {
            let _ = writeln!(out, "    {:>9.1} MiB {name}: {} x {}", mib(k.upload_causes[cause]), k.dispatches, k.name);
        }
    }
    let mut movers: Vec<&KernelProfile> = runtime.profile.iter().collect();
    movers.sort_by_key(|k| std::cmp::Reverse(k.copied_in + k.copied_out + k.uploaded + k.downloaded));
    for k in movers.iter().take(15).filter(|k| k.copied_in + k.copied_out + k.uploaded + k.downloaded > 0) {
        let _ = writeln!(
            out,
            "  {:>10.1} MiB copied in {:>10.1} out, {:>10.1} uploaded {:>10.1} downloaded  {:>8} x  {}",
            mib(k.copied_in),
            mib(k.copied_out),
            mib(k.uploaded),
            mib(k.downloaded),
            k.dispatches,
            k.name
        );
    }
    // `CLEAVE_VK_PROFILE_KERNELS`: how many kernels to list, 30 by default.
    let listed = std::env::var("CLEAVE_VK_PROFILE_KERNELS").ok().and_then(|v| v.parse().ok()).unwrap_or(30);
    for k in kernels.iter().take(listed) {
        let mean = if k.timed > 0 { k.gpu_ns / k.timed as f64 / 1e3 } else { 0.0 };
        let _ = writeln!(
            out,
            "  {:>9.1} ms {:>5.1}%  {:>8} x {:>8.1} us  {}",
            k.gpu_ns / 1e6,
            100.0 * k.gpu_ns / total.max(1.0),
            k.dispatches,
            mean,
            k.name
        );
    }
    // Where the device's time goes beyond the costliest kernels: by the
    // function a kernel was outlined from, and by how long its dispatches
    // last (what fusing small kernels could save). A kernel's time estimated
    // from its timed dispatches' mean.
    let estimate = |k: &KernelProfile| if k.timed > 0 { k.gpu_ns / k.timed as f64 * k.dispatches as f64 } else { 0.0 };
    let all: f64 = runtime.profile.iter().map(estimate).sum::<f64>().max(1.0);
    let mut functions: std::collections::HashMap<&str, (f64, u64, usize)> = Default::default();
    let mut lengths = [(0.0, 0u64, 0usize); 4];
    const LENGTHS: [(&str, f64); 4] = [("under 5 us", 5.0), ("5-20 us", 20.0), ("20-100 us", 100.0), ("100 us and up", f64::MAX)];
    for k in &runtime.profile {
        let ns = estimate(k);
        let function = k.name.split("_kernel").next().unwrap_or(&k.name);
        let entry = functions.entry(function).or_default();
        entry.0 += ns;
        entry.1 += k.dispatches;
        entry.2 += 1;
        let mean = if k.timed > 0 { k.gpu_ns / k.timed as f64 / 1e3 } else { 0.0 };
        let bucket = LENGTHS.iter().position(|(_, below)| mean < *below).unwrap_or(3);
        lengths[bucket].0 += ns;
        lengths[bucket].1 += k.dispatches;
        lengths[bucket].2 += 1;
    }
    let _ = writeln!(out, "  device time by dispatch length:");
    for ((label, _), (ns, dispatches, count)) in LENGTHS.iter().zip(lengths) {
        let _ = writeln!(
            out,
            "  {:>9.1} ms {:>5.1}%  {:>9} dispatches of {:>5} kernels  {label}",
            ns / 1e6,
            100.0 * ns / all,
            dispatches,
            count
        );
    }
    let mut functions: Vec<_> = functions.into_iter().collect();
    functions.sort_by(|a, b| b.1.0.total_cmp(&a.1.0));
    let _ = writeln!(out, "  device time by function:");
    for (function, (ns, dispatches, count)) in functions.iter().take(25) {
        let _ = writeln!(
            out,
            "  {:>9.1} ms {:>5.1}%  {:>9} dispatches of {:>5} kernels  {function}",
            ns / 1e6,
            100.0 * ns / all,
            dispatches,
            count
        );
    }
    eprint!("{out}");
}

fn lock() -> std::sync::MutexGuard<'static, Option<Runtime>> {
    RUNTIME.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// What the runtime has moved so far.
pub fn stats() -> Stats {
    lock().as_ref().map(|r| r.stats).unwrap_or_default()
}

/// Waits for every command recorded or submitted so far. Generated code
/// never needs it (a host read waits for what it reads); for a host that
/// wants the device idle.
pub fn finish() {
    if let Some(runtime) = lock().as_mut() {
        runtime.finish();
    }
}

impl Runtime {
    /// The command buffer being recorded, begun if none is, ready for a
    /// command reading `reads` and writing `writes`: a memory barrier first
    /// when it needs one, a buffer it touches written by a command since the
    /// last barrier, or one it writes read by one (or the batch's first
    /// command: an earlier batch may still run). Commands on unrelated
    /// buffers then run without draining the device between them (30% of
    /// nanoLM's barriers needed nothing). `CLEAVE_VK_FULL_BARRIERS=1`: one
    /// before every command, a diagnostic.
    fn record(&mut self, reads: &[vk::Buffer], writes: &[vk::Buffer]) -> vk::CommandBuffer {
        let device = &self.gpu.device;
        if self.recording.is_none() {
            let command = unsafe {
                device.allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(self.gpu.command_pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
            }
            .unwrap_or_else(|e| abort_with(format!("allocating a command buffer: {e}")))[0];
            let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }
                .unwrap_or_else(|e| abort_with(format!("creating a fence: {e}")));
            unsafe {
                device.begin_command_buffer(
                    command,
                    &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
            }
            .unwrap_or_else(|e| abort_with(format!("beginning a command buffer: {e}")));
            let queries = self.profiling.then(|| {
                let pool = unsafe {
                    device.create_query_pool(
                        &vk::QueryPoolCreateInfo::default()
                            .query_type(vk::QueryType::TIMESTAMP)
                            .query_count(2 * TIMED_PER_BATCH as u32),
                        None,
                    )
                }
                .unwrap_or_else(|e| abort_with(format!("creating a query pool: {e}")));
                unsafe { device.cmd_reset_query_pool(command, pool, 0, 2 * TIMED_PER_BATCH as u32) };
                pool
            });
            self.recording = Some(Batch {
                command,
                fence,
                commands: 0,
                garbage: Vec::new(),
                queries,
                timed: Vec::new(),
                read_since: Default::default(),
                written_since: Default::default(),
            });
        }
        let batch = self.recording.as_mut().expect("a batch");
        let needed = full_barriers()
            || batch.commands == 0
            || reads.iter().chain(writes).any(|b| batch.written_since.contains(b))
            || writes.iter().any(|b| batch.read_since.contains(b));
        self.stats.barriers[needed as usize] += 1;
        if needed {
            batch.read_since.clear();
            batch.written_since.clear();
            // A barrier's first scope is everything submitted to the queue
            // before it: the commands of this batch and of earlier ones.
            let stages = vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER;
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(
                    vk::AccessFlags::SHADER_READ
                        | vk::AccessFlags::SHADER_WRITE
                        | vk::AccessFlags::TRANSFER_READ
                        | vk::AccessFlags::TRANSFER_WRITE,
                );
            unsafe {
                device.cmd_pipeline_barrier(batch.command, stages, stages, vk::DependencyFlags::empty(), &[barrier], &[], &[])
            };
        }
        batch.read_since.extend(reads);
        batch.written_since.extend(writes);
        batch.commands += 1;
        batch.command
    }

    /// Keeps `garbage` alive until the device has finished everything that
    /// may use it: the batch being recorded, else the last one in flight;
    /// dropped at once when nothing is pending.
    fn defer(&mut self, garbage: Garbage) {
        // A buffer only the device uses is spare at once: what's recorded
        // with it next runs after what used it (one queue, a barrier before
        // any command touching a buffer an unfinished one touched). Waiting for the fence left the spare empty
        // between waits, every new mirror a new buffer (320k a nanoLM step).
        // One the host maps waits: the host writes it when it's taken.
        if let Garbage::Buffer(buffer) = &garbage
            && buffer.mapped.is_null()
        {
            self.dispose(garbage);
            return;
        }
        if let Some(batch) = self.recording.as_mut().or(self.in_flight.last_mut()) {
            batch.garbage.push(garbage);
        } else {
            self.dispose(garbage);
        }
    }

    /// A buffer of `len` bytes, mapped for the host (`host`) or in the
    /// device's own memory: a spare one if there is (`spare`), else a new one.
    fn take_buffer(&mut self, len: usize, host: bool, pinned: &[usize]) -> Buffer<'static> {
        if let Some(buffer) = self.spare.get_mut(&(len, host)).and_then(Vec::pop) {
            self.spare_bytes[host as usize] -= len;
            return buffer;
        }
        if !host {
            self.make_room(len, pinned);
            if self.profiling
                && let Some([used, budget, blocks, allocations]) = self.gpu.device_memory_detail()
                && used > self.stats.device_peak[0]
            {
                let mirrors: usize = self.mirrors.values().map(|m| m.len).sum();
                self.stats.device_peak =
                    [used, budget, mirrors as u64, self.spare_bytes[0] as u64, blocks, allocations];
            }
        }
        self.stats.buffers_made += 1;
        let make = |gpu: &'static Gpu| if host { gpu.buffer(len) } else { gpu.device_buffer(len) };
        match make(self.gpu) {
            Ok(buffer) => buffer,
            // Out of memory, perhaps for the spare buffers held (nanoLM at its
            // full size, 12 GiB of device memory): the device's work finished
            // (what it holds freed or made spare), every spare freed, then once
            // more.
            Err(_) => {
                self.finish();
                self.spare.clear();
                self.spare_bytes = [0, 0];
                self.spare_order = Default::default();
                loop {
                    match make(self.gpu) {
                        Ok(buffer) => return buffer,
                        Err(e) if host => abort_with(format!("a buffer of {len} bytes: {e}")),
                        Err(e) => {
                            // The least recently used mirror the dispatch being
                            // prepared doesn't bind, its device copy brought home
                            // first if newer.
                            let victim = self
                                .mirrors
                                .iter()
                                .filter(|(key, _)| !pinned.contains(key))
                                .min_by_key(|(_, mirror)| mirror.last_use)
                                .map(|(&key, _)| key);
                            let Some(key) = victim else {
                                abort_with(format!("a buffer of {len} bytes, every mirror evicted: {e}"))
                            };
                            self.evict(key);
                        }
                    }
                }
            }
        }
    }

    /// Whether `len` more bytes of the device's memory would take the process
    /// past what it may use: `CLEAVE_VK_DEVICE_BUDGET` bytes of mirrors (what
    /// tests it, on a small program), or 95% of the system's budget for it
    /// (`Gpu::device_memory`: past the budget, Windows doesn't fail an
    /// allocation but moves device memory to the host's behind the device's
    /// back, nanoLM's step 3x slower beside a game).
    fn over_budget(&self, len: usize) -> bool {
        if let Some(budget) = device_budget() {
            let mirrors: usize = self.mirrors.values().map(|m| m.len).sum();
            if mirrors + len > budget {
                return true;
            }
        }
        self.gpu.device_memory().is_some_and(|(used, budget)| used + len as u64 > budget / 20 * 19)
    }

    /// The spare device buffers freed (`spare`) once the device has finished
    /// what's recorded: one is spare as soon as its last command is recorded
    /// (`defer`).
    fn free_spare_device_buffers(&mut self) {
        let device: Vec<(usize, bool)> = self.spare.keys().filter(|(_, host)| !host).copied().collect();
        for key in device {
            for buffer in self.spare.remove(&key).unwrap_or_default() {
                self.defer(Garbage::Free(buffer));
            }
        }
        self.spare_bytes[0] = 0;
        self.spare_order[0].clear();
    }

    /// Room for `len` more bytes of the device's memory (`over_budget`): the
    /// device's work finished, the spare buffers freed, then the least
    /// recently used mirrors the dispatch being prepared doesn't bind
    /// (`pinned`) evicted, until it fits or nothing more can go.
    fn make_room(&mut self, len: usize, pinned: &[usize]) {
        if !self.over_budget(len) {
            return;
        }
        self.finish();
        self.free_spare_device_buffers();
        while self.over_budget(len) {
            let victim = self
                .mirrors
                .iter()
                .filter(|(key, _)| !pinned.contains(key))
                .min_by_key(|(_, mirror)| mirror.last_use)
                .map(|(&key, _)| key);
            let Some(key) = victim else { break };
            self.evict(key);
            // Its buffer freed, not kept spare.
            self.free_spare_device_buffers();
        }
    }

    /// Frees the mirror at `key`, the device being idle (`finish`): its
    /// contents brought home if the device wrote them since; the next
    /// dispatch that binds the payload uploads it again.
    fn evict(&mut self, key: usize) {
        self.download(&[key]);
        if let Some(mirror) = self.mirrors.remove(&key) {
            drop(mirror.buffer);
            self.stats.evictions += 1;
            bump(&DEVICE_GENERATION);
            bump(&HOST_GENERATION);
        }
    }

    fn dispose(&mut self, garbage: Garbage) {
        match garbage {
            Garbage::Buffer(buffer) => {
                let (len, host) = (buffer.len(), !buffer.mapped.is_null());
                let kind = host as usize;
                let limit = if host { SPARE_HOST_BYTES } else { SPARE_DEVICE_BYTES };
                if len > limit {
                    return;
                }
                // The oldest kept go until this one fits.
                while self.spare_bytes[kind] + len > limit {
                    let Some(oldest) = self.spare_order[kind].pop_front() else { break };
                    if let Some(kept) = self.spare.get_mut(&(oldest, host)).filter(|kept| !kept.is_empty()) {
                        // Perhaps still bound by a command in flight.
                        let pushed_out = kept.remove(0);
                        self.spare_bytes[kind] -= oldest;
                        self.defer(Garbage::Free(pushed_out));
                    }
                }
                self.spare_bytes[kind] += len;
                self.spare.entry((len, host)).or_default().push(buffer);
                self.spare_order[kind].push_back(len);
                // Entries of buffers taken since pile up: rebuilt from what's
                // kept once they far outnumber it (the order within a size
                // and across sizes then approximate).
                if self.spare_order[kind].len() > 65536 {
                    self.spare_order[kind] = self
                        .spare
                        .iter()
                        .filter(|((_, h), _)| *h == host)
                        .flat_map(|((size, _), kept)| std::iter::repeat_n(*size, kept.len()))
                        .collect();
                }
            }
            Garbage::DescriptorPool(pool) => unsafe { self.gpu.device.destroy_descriptor_pool(pool, None) },
            Garbage::Free(buffer) => drop(buffer),
        }
    }

    /// Submits the batch being recorded, without waiting for it.
    fn submit(&mut self) {
        let Some(batch) = self.recording.take() else { return };
        let device = &self.gpu.device;
        unsafe {
            // The batch's writes made visible to the host's reads once its
            // fence signals (downloads land in host-visible staging buffers).
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ);
            device.cmd_pipeline_barrier(
                batch.command,
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
            device.end_command_buffer(batch.command).unwrap_or_else(|e| abort_with(format!("ending a command buffer: {e}")));
            let commands = [batch.command];
            device
                .queue_submit(self.gpu.queue, &[vk::SubmitInfo::default().command_buffers(&commands)], batch.fence)
                .unwrap_or_else(|e| abort_with(format!("submitting a batch: {e}")));
        }
        self.stats.submits += 1;
        self.in_flight.push(batch);
        // What the device has finished released now, not at the host's next
        // wait: the buffers the batches free (spare ones pushed out) piled
        // up between waits, 3.7 GiB of nanoLM's device memory. At most
        // `MOST_IN_FLIGHT` batches ahead of the device, the oldest waited
        // for past that.
        self.retire_finished();
        if self.in_flight.len() > MOST_IN_FLIGHT {
            let waiting = std::time::Instant::now();
            let fence = self.in_flight[0].fence;
            unsafe { self.gpu.device.wait_for_fences(&[fence], true, u64::MAX) }
                .unwrap_or_else(|e| abort_with(format!("waiting for a batch: {e}")));
            self.stats.waits += 1;
            self.stats.wait_ns += waiting.elapsed().as_nanos() as u64;
            self.retire_finished();
        }
        // The system's budget read again, the spare buffers given back when
        // it has shrunk under what the process uses (another program took
        // device memory); `take_buffer` evicts mirrors if that isn't enough.
        self.gpu.refresh_budget(self.stats.submits as u32);
        if self.over_budget(0) {
            self.free_spare_device_buffers();
        }
    }

    /// Submits what is recorded and waits for everything submitted, then
    /// releases what the finished batches kept alive.
    fn finish(&mut self) {
        self.submit();
        if self.in_flight.is_empty() {
            return;
        }
        let device = &self.gpu.device;
        let waiting = std::time::Instant::now();
        for batch in &self.in_flight {
            unsafe { device.wait_for_fences(&[batch.fence], true, u64::MAX) }
                .unwrap_or_else(|e| abort_with(format!("waiting for a batch: {e}")));
        }
        self.stats.waits += 1;
        self.stats.wait_ns += waiting.elapsed().as_nanos() as u64;
        for batch in std::mem::take(&mut self.in_flight) {
            self.retire(batch);
        }
        for message in self.gpu.validation_messages() {
            eprintln!("cleave Vulkan validation: {message}");
        }
    }

    /// The batches the device has finished, from the oldest, released
    /// (`retire`); none waited for.
    fn retire_finished(&mut self) {
        while let Some(batch) = self.in_flight.first() {
            if !unsafe { self.gpu.device.get_fence_status(batch.fence) }.unwrap_or(false) {
                break;
            }
            let batch = self.in_flight.remove(0);
            self.retire(batch);
        }
    }

    /// A batch the device has finished: its timings read, its objects
    /// destroyed, what it kept alive released (`dispose`).
    fn retire(&mut self, batch: Batch) {
        let device = &self.gpu.device;
        if let Some(pool) = batch.queries {
            if !batch.timed.is_empty() {
                let mut ticks = vec![0u64; 2 * batch.timed.len()];
                let read = unsafe {
                    device.get_query_pool_results(pool, 0, &mut ticks, vk::QueryResultFlags::TYPE_64)
                };
                if read.is_ok() {
                    for (i, &kernel) in batch.timed.iter().enumerate() {
                        let ns = ticks[2 * i + 1].saturating_sub(ticks[2 * i]) as f64 * self.gpu.timestamp_period as f64;
                        self.profile[kernel].gpu_ns += ns;
                        self.profile[kernel].timed += 1;
                    }
                    // The batch's span, from its first kernel's start to
                    // its last's end, against the union of its kernels'
                    // intervals: what's left is the device between
                    // kernels (barriers draining it, launches, copies).
                    let mut intervals: Vec<(u64, u64)> =
                        ticks.chunks(2).map(|t| (t[0], t[1].max(t[0]))).collect();
                    intervals.sort_unstable();
                    let (first, last) = (intervals[0].0, intervals.iter().map(|i| i.1).max().unwrap_or(0));
                    let mut busy = 0u64;
                    let mut current = intervals[0];
                    for &(start, end) in &intervals[1..] {
                        if start > current.1 {
                            busy += current.1 - current.0;
                            current = (start, end);
                        } else {
                            current.1 = current.1.max(end);
                        }
                    }
                    busy += current.1 - current.0;
                    let period = self.gpu.timestamp_period as f64;
                    self.stats.batch_span_ns += ((last - first) as f64 * period) as u64;
                    self.stats.batch_busy_ns += (busy as f64 * period) as u64;
                }
            }
            unsafe { device.destroy_query_pool(pool, None) };
        }
        unsafe {
            device.destroy_fence(batch.fence, None);
            device.free_command_buffers(self.gpu.command_pool, &[batch.command]);
        }
        for garbage in batch.garbage {
            self.dispose(garbage);
        }
    }

    /// The mirrors overlapping `[start, start + len)`, by start address.
    fn overlapping(&self, start: usize, len: usize) -> Vec<usize> {
        let end = start + len.max(1);
        let mut found = Vec::new();
        // The one starting before `start` may reach into the range.
        if let Some((&key, mirror)) = self.mirrors.range(..start).next_back() {
            if key + mirror.len > start {
                found.push(key);
            }
        }
        found.extend(self.mirrors.range(start..end).map(|(&key, _)| key));
        found
    }

    /// Brings the host copies of the mirrors at `keys` that the device wrote
    /// up to date: one copy each into a host-visible staging buffer, one wait
    /// for all of them.
    fn download(&mut self, keys: &[usize]) {
        let stale: Vec<usize> = keys.iter().copied().filter(|k| self.mirrors[k].device_newer).collect();
        if stale.is_empty() {
            return;
        }
        let gpu = self.gpu;
        let mut staged = Vec::with_capacity(stale.len());
        for &key in &stale {
            let len = self.mirrors[&key].len;
            let staging = self.take_buffer(len, true, &[]);
            let command = self.record(&[self.mirrors[&key].buffer.buffer], &[staging.buffer]);
            let region = [vk::BufferCopy::default().size(len as u64)];
            unsafe { gpu.device.cmd_copy_buffer(command, self.mirrors[&key].buffer.buffer, staging.buffer, &region) };
            staged.push((key, staging));
        }
        self.finish();
        for (key, staging) in staged {
            let mirror = self.mirrors.get_mut(&key).expect("a mirror");
            // SAFETY: the mirror's host range is a live payload (a freed one's
            // mirror is gone, `on_freed`).
            let host = unsafe { std::slice::from_raw_parts_mut(key as *mut u8, mirror.len) };
            let copying = std::time::Instant::now();
            staging.read(host);
            mirror.device_newer = false;
            let len = mirror.len;
            self.stats.download_copy_ns += copying.elapsed().as_nanos() as u64;
            self.stats.downloaded_bytes += len as u64;
            self.dispose(Garbage::Buffer(staging)); // the device is done with it
            if tracing() {
                eprintln!("cleave Vulkan download {key:#x}+{len}");
            }
        }
    }

    /// Records an upload of the host's `[start, start + len)` into `target`:
    /// the bytes copied now into a fresh staging buffer, which nothing in
    /// flight uses, and copied on the device in order with what's recorded.
    fn upload(&mut self, start: usize, len: usize, target: vk::Buffer) {
        let entered = std::time::Instant::now();
        let gpu = self.gpu;
        let mut staging = self.take_buffer(len, true, &[]);
        // SAFETY: the caller's range is live host memory.
        staging.write(unsafe { std::slice::from_raw_parts(start as *const u8, len) });
        let command = self.record(&[staging.buffer], &[target]);
        let region = [vk::BufferCopy::default().size(len as u64)];
        unsafe { gpu.device.cmd_copy_buffer(command, staging.buffer, target, &region) };
        self.defer(Garbage::Buffer(staging));
        self.stats.uploaded_bytes += len as u64;
        self.stats.upload_ns += entered.elapsed().as_nanos() as u64;
        if tracing() {
            eprintln!("cleave Vulkan upload {start:#x}+{len}");
        }
    }

    /// Makes the mirror of exactly `[start, start + len)` up to date with the
    /// host copy: created and uploaded, or uploaded again if the host wrote
    /// it. Mirrors overlapping it otherwise are retired, their device writes
    /// brought home first. `discard`: what's there is about to be overwritten
    /// whole ([`cleave_vk_discard`]), so nothing is uploaded, nor brought home
    /// from a mirror inside the range.
    fn mirror(&mut self, start: usize, len: usize, discard: bool, pinned: &[usize]) {
        let retired: Vec<usize> = self
            .overlapping(start, len)
            .into_iter()
            .filter(|&key| key != start || self.mirrors[&key].len != len)
            .collect();
        let kept: Vec<usize> = retired
            .iter()
            .copied()
            .filter(|&key| !(discard && key >= start && key + self.mirrors[&key].len <= start + len))
            .collect();
        self.download(&kept);
        for key in retired {
            let mirror = self.mirrors.remove(&key).expect("a mirror");
            self.defer(Garbage::Buffer(mirror.buffer));
            bump(&DEVICE_GENERATION);
            bump(&HOST_GENERATION);
        }
        self.uses += 1;
        if !self.mirrors.contains_key(&start) {
            let buffer = self.take_buffer(len, false, pinned);
            self.mirrors.insert(start, Mirror { len, buffer, host_dirty: true, device_newer: false, last_use: self.uses });
            MIRRORS_LIVE.store(true, std::sync::atomic::Ordering::Release);
            bump(&DEVICE_GENERATION);
            bump(&HOST_GENERATION);
        }
        let mirror = self.mirrors.get_mut(&start).expect("a mirror");
        mirror.last_use = self.uses;
        if discard {
            mirror.host_dirty = false;
            mirror.device_newer = false;
            bump(&DEVICE_GENERATION);
            bump(&HOST_GENERATION);
        } else if mirror.host_dirty {
            mirror.host_dirty = false;
            bump(&HOST_GENERATION);
            let target = mirror.buffer.buffer;
            self.upload(start, len, target);
        }
    }
}

/// `cleave_rt`'s free hook: a payload range stops being live, its mirrors go
/// (without a download: nothing reads a dead payload), their buffers kept
/// until the device has finished what may use them.
extern "C" fn on_freed(start: *mut u8, len: usize) {
    if !MIRRORS_LIVE.load(std::sync::atomic::Ordering::Acquire) {
        return;
    }
    if tracing() {
        eprintln!("cleave Vulkan freed {:#x}+{len}", start as usize);
    }
    let mut guard = lock();
    if let Some(runtime) = guard.as_mut() {
        for key in runtime.overlapping(start as usize, len) {
            let mirror = runtime.mirrors.remove(&key).expect("a mirror");
            runtime.defer(Garbage::Buffer(mirror.buffer));
            bump(&DEVICE_GENERATION);
            bump(&HOST_GENERATION);
        }
        MIRRORS_LIVE.store(!runtime.mirrors.is_empty(), std::sync::atomic::Ordering::Release);
    }
}

/// A kernel argument's [`cleave_vk_dispatch`] flag: its memory is a heap
/// payload (`cleave_alloc_rc`, whose frees `cleave_rt` reports), which may
/// therefore stay mirrored on the device across dispatches.
pub const ARG_MIRROR: i64 = 1;
/// A kernel argument's flag: the kernel writes it.
pub const ARG_WRITTEN: i64 = 2;
/// A kernel argument's flag: the kernel reads it (informative: what a kernel
/// overwrites whole is said by [`cleave_vk_discard`]).
pub const ARG_READ: i64 = 4;

/// One kernel launch from generated code (`cleave-lower-gpu-launches`): the
/// entry point `entry` (a NUL-terminated name) of the SPIR-V binary `spirv`
/// (`words` 32-bit words), over `groups_x * groups_y * groups_z`
/// workgroups, its `count` storage buffers the host memory `pointers[i]`,
/// `bytes[i]` bytes each, flagged `flags[i]` ([`ARG_MIRROR`],
/// [`ARG_WRITTEN`], [`ARG_READ`]), in binding order. Every argument is an
/// `i64`, the pointers included.
///
/// Recorded, not waited for (`doc/plan-vulkan.md`, Phase 2): a mirrored
/// argument is uploaded only if the host wrote it since
/// ([`cleave_vk_host_wrote`]), and what the kernel writes stays on the
/// device until the host asks for it ([`cleave_vk_sync_host`]). Any other
/// argument is copied in, and, if the kernel writes it, back before this
/// returns (its memory may not outlive the caller's frame), which waits.
/// A failure aborts the process with a message: generated code has no
/// error path to return one through.
///
/// # Safety
///
/// `spirv`, `entry`, `pointers`, `bytes` and `flags` point to what is
/// described above, and each `pointers[i]` to `bytes[i]` bytes the caller
/// owns, a live payload when flagged [`ARG_MIRROR`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cleave_vk_dispatch(
    spirv: i64,
    words: i64,
    entry: i64,
    groups_x: i64,
    groups_y: i64,
    groups_z: i64,
    count: i64,
    pointers: i64,
    bytes: i64,
    flags: i64,
) {
    // SAFETY: as documented, every pointer is valid for what it describes.
    let entry = unsafe { CStr::from_ptr(entry as usize as *const std::ffi::c_char) }.to_string_lossy().into_owned();
    let count = count as usize;
    let (pointers, bytes, flags) = unsafe {
        (
            std::slice::from_raw_parts(pointers as usize as *const i64, count),
            std::slice::from_raw_parts(bytes as usize as *const i64, count),
            std::slice::from_raw_parts(flags as usize as *const i64, count),
        )
    };
    // SAFETY: `spirv` points to `words` words, as documented.
    let binary = unsafe { std::slice::from_raw_parts(spirv as usize as *const u32, words as usize) };
    unsafe { dispatch(binary, &entry, &entry, &[], [groups_x, groups_y, groups_z], pointers, bytes, flags) };
}

/// The bfloat16 GEMM kernel (`kernels/gemm_bf16.comp`), compiled by `kernels/build.ps1`.
static GEMM_BF16: &[u8] = include_bytes!("../kernels/gemm_bf16.spv");

/// A product the program asked to compute in bfloat16 (`stdlib/linalg`'s `MatMulBf16`, lowered by
/// `cleave-lower-gpu-gemms`): `C = op(A) op(B)` with `A`, `B`, `C` float32 heap payloads (`a`, `b`,
/// `c`, `*_bytes` bytes each, mirrored as a kernel's arguments are), `A` `m x k` (`k x m` stored
/// when `ta`), `B` `k x n` (`n x k` when `tb`), on the tensor cores (`kernels/gemm_bf16.comp`). `m`
/// and `n` multiples of 128, `k` of 32: the compiler checks. Recorded like a dispatch, not waited for.
///
/// # Safety
///
/// Each pointer is a live heap payload of the given size, as for [`cleave_vk_dispatch`]'s mirrored
/// arguments.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cleave_vk_gemm_bf16(
    a: i64,
    a_bytes: i64,
    b: i64,
    b_bytes: i64,
    c: i64,
    c_bytes: i64,
    m: i64,
    n: i64,
    k: i64,
    ta: i64,
    tb: i64,
) {
    static WORDS: std::sync::OnceLock<Vec<u32>> = std::sync::OnceLock::new();
    let binary = WORDS.get_or_init(|| {
        GEMM_BF16.chunks_exact(4).map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]])).collect()
    });
    if m % 128 != 0 || n % 128 != 0 || k % 32 != 0 {
        abort_with(format!("cleave_vk_gemm_bf16: {m}x{n}x{k} is not a multiple of 128x128x32"));
    }
    let (lda, ldb, ldc) = (if ta != 0 { m } else { k }, if tb != 0 { k } else { n }, n);
    let constants = [
        (0, m as u32),
        (1, n as u32),
        (2, k as u32),
        (3, lda as u32),
        (4, ldb as u32),
        (5, ldc as u32),
        (6, (ta != 0) as u32),
        (7, (tb != 0) as u32),
    ];
    let name = format!("gemm_bf16<{m}x{n}x{k}{}{}>", if ta != 0 { ", A^T" } else { "" }, if tb != 0 { ", B^T" } else { "" });
    let pointers = [a, b, c];
    let bytes = [a_bytes, b_bytes, c_bytes];
    let flags = [ARG_MIRROR | ARG_READ, ARG_MIRROR | ARG_READ, ARG_MIRROR | ARG_WRITTEN];
    unsafe { dispatch(binary, "main", &name, &constants, [n / 128, m / 128, 1], &pointers, &bytes, &flags) };
}

/// The bfloat16-operand GEMM kernel (`kernels/gemm_bf16_in.comp`).
static GEMM_BF16_OPERANDS: &[u8] = include_bytes!("../kernels/gemm_bf16_in.spv");

/// A product of bf16 tensors (`cleave-lower-gpu-gemms`): as [`cleave_vk_gemm_bf16`], with `a` and `b`
/// bfloat16 in memory (`kernels/gemm_bf16_in.comp`, no rounding on the way), `c` float32. `m` and
/// `n` multiples of 128, `k` of 32, every leading dimension of 8: the compiler checks.
///
/// # Safety
///
/// As [`cleave_vk_gemm_bf16`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cleave_vk_gemm_bf16_operands(
    a: i64,
    a_bytes: i64,
    b: i64,
    b_bytes: i64,
    c: i64,
    c_bytes: i64,
    m: i64,
    n: i64,
    k: i64,
    ta: i64,
    tb: i64,
) {
    static WORDS: std::sync::OnceLock<Vec<u32>> = std::sync::OnceLock::new();
    let binary = WORDS.get_or_init(|| {
        GEMM_BF16_OPERANDS.chunks_exact(4).map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]])).collect()
    });
    if m % 128 != 0 || n % 128 != 0 || k % 32 != 0 {
        abort_with(format!("cleave_vk_gemm_bf16_operands: {m}x{n}x{k} is not a multiple of 128x128x32"));
    }
    let (lda, ldb, ldc) = (if ta != 0 { m } else { k }, if tb != 0 { k } else { n }, n);
    let constants = [
        (0, m as u32),
        (1, n as u32),
        (2, k as u32),
        (3, lda as u32),
        (4, ldb as u32),
        (5, ldc as u32),
        (6, (ta != 0) as u32),
        (7, (tb != 0) as u32),
    ];
    let name = format!(
        "gemm_bf16_operands<{m}x{n}x{k}{}{}>",
        if ta != 0 { ", A^T" } else { "" },
        if tb != 0 { ", B^T" } else { "" }
    );
    let pointers = [a, b, c];
    let bytes = [a_bytes, b_bytes, c_bytes];
    let flags = [ARG_MIRROR | ARG_READ, ARG_MIRROR | ARG_READ, ARG_MIRROR | ARG_WRITTEN];
    unsafe { dispatch(binary, "main", &name, &constants, [n / 128, m / 128, 1], &pointers, &bytes, &flags) };
}

/// One kernel launch: `cleave_vk_dispatch`'s work, for any kernel. `entry` is the binary's entry
/// point, `key` how the kernel is cached and profiled (with `constants`, its specialization: the
/// same binary specialized differently is another kernel).
///
/// # Safety
///
/// As [`cleave_vk_dispatch`]: each `pointers[i]` to `bytes[i]` bytes the caller owns, a live
/// payload when flagged [`ARG_MIRROR`].
unsafe fn dispatch(
    binary: &[u32],
    entry: &str,
    key_name: &str,
    constants: &[(u32, u32)],
    [groups_x, groups_y, groups_z]: [i64; 3],
    pointers: &[i64],
    bytes: &[i64],
    flags: &[i64],
) {
    let entered = std::time::Instant::now();
    let mut guard = lock();
    let runtime = runtime(&mut guard);
    let gpu = runtime.gpu;
    // Ends phase `$i` (`Stats::dispatch_phase_ns`), the waits in it left out.
    let (mut phase_start, mut phase_waited) = (entered, runtime.stats.wait_ns);
    macro_rules! phase {
        ($i:expr) => {{
            let now = std::time::Instant::now();
            let waited = runtime.stats.wait_ns - phase_waited;
            runtime.stats.dispatch_phase_ns[$i] += ((now - phase_start).as_nanos() as u64).saturating_sub(waited);
            phase_start = now;
            phase_waited = runtime.stats.wait_ns;
        }};
    }
    let count = pointers.len();
    if tracing() {
        let args: Vec<String> =
            (0..count).map(|i| format!("{:#x}+{} f{}", pointers[i], bytes[i], flags[i])).collect();
        eprintln!("cleave Vulkan dispatch `{entry}` [{groups_x}, {groups_y}, {groups_z}]: {}", args.join(", "));
    }
    let key = (binary.as_ptr() as usize, key_name.to_string());
    if !runtime.kernels.contains_key(&key) {
        let kernel = gpu
            .kernel_specialized(binary, entry, count as u32, constants)
            .unwrap_or_else(|e| abort_with(format!("kernel `{key_name}`: {e}")));
        runtime.kernels.insert(key.clone(), kernel);
        let id = runtime.profile.len();
        runtime.profile.push(KernelProfile {
            name: format!("{key_name} #{id} [{groups_x}, {groups_y}, {groups_z}]"),
            ..Default::default()
        });
        runtime.kernel_ids.insert(key.clone(), id);
    }
    let kernel_id = runtime.kernel_ids[&key];
    runtime.profile[kernel_id].dispatches += 1;
    phase!(0);
    let (uploaded_before, downloaded_before) = (runtime.stats.uploaded_bytes, runtime.stats.downloaded_bytes);

    if paranoid().0 {
        // Not over a newer device copy: that one is the truth.
        for mirror in runtime.mirrors.values_mut().filter(|m| !m.device_newer) {
            mirror.host_dirty = true;
        }
    }
    // The mirrors first (each may retire another), then the copies. None of
    // this dispatch's is evicted to make room for another (`pinned`).
    let pinned: Vec<usize> =
        (0..count).filter(|&i| flags[i] & ARG_MIRROR != 0).map(|i| pointers[i] as usize).collect();
    for i in 0..count {
        if flags[i] & ARG_MIRROR != 0 {
            let (start, len) = (pointers[i] as usize, bytes[i] as usize);
            let new = !runtime.mirrors.get(&start).is_some_and(|m| m.len == len);
            let before = runtime.stats.uploaded_bytes;
            runtime.mirror(start, len, false, &pinned);
            let uploaded = runtime.stats.uploaded_bytes - before;
            if uploaded > 0 {
                let cause = match (new, flags[i] & ARG_READ != 0) {
                    (true, false) => 0,
                    (true, true) => 1,
                    (false, _) => 2,
                };
                runtime.stats.upload_causes[cause] += uploaded;
                runtime.profile[kernel_id].upload_causes[cause] += uploaded;
            }
        }
    }
    // A copied argument may still be memory a mirror covers (a payload the
    // compiler proved elsewhere and not here, a public function's parameter):
    // the host copy brought up to date before it's read.
    for i in 0..count {
        if flags[i] & ARG_MIRROR == 0 {
            let keys = runtime.overlapping(pointers[i] as usize, bytes[i] as usize);
            runtime.download(&keys);
        }
    }
    {
        let profile = &mut runtime.profile[kernel_id];
        profile.uploaded += runtime.stats.uploaded_bytes - uploaded_before;
        profile.downloaded += runtime.stats.downloaded_bytes - downloaded_before;
    }
    phase!(1);
    let mut copies: Vec<Option<Buffer<'static>>> = Vec::with_capacity(count);
    for i in 0..count {
        if flags[i] & ARG_MIRROR != 0 {
            copies.push(None);
            continue;
        }
        let mut buffer =
            runtime.take_buffer(bytes[i] as usize, true, &[]);
        // Copied in whether the kernel reads it or not: one that writes only
        // part of it would bring the rest back as garbage otherwise.
        let host = unsafe { std::slice::from_raw_parts(pointers[i] as usize as *const u8, bytes[i] as usize) };
        buffer.write(host);
        runtime.stats.uploaded_bytes += bytes[i] as u64;
        runtime.profile[kernel_id].copied_in += bytes[i] as u64;
        copies.push(Some(buffer));
    }
    phase!(2);
    let buffers: Vec<vk::Buffer> = (0..count)
        .map(|i| match &copies[i] {
            Some(buffer) => buffer.buffer,
            None => runtime.mirrors[&(pointers[i] as usize)].buffer.buffer,
        })
        .collect();

    // The buffers pushed with the dispatch, or a descriptor set from a pool
    // of its own, kept until the batch is done.
    let kernel = &runtime.kernels[&key];
    let (pipeline, pipeline_layout, set_layout, pushed) =
        (kernel.pipeline, kernel.pipeline_layout, kernel.set_layout, kernel.pushed);
    let device = &gpu.device;
    let infos: Vec<[vk::DescriptorBufferInfo; 1]> = buffers
        .iter()
        .map(|&b| [vk::DescriptorBufferInfo::default().buffer(b).offset(0).range(vk::WHOLE_SIZE)])
        .collect();
    let mut writes: Vec<vk::WriteDescriptorSet> = infos
        .iter()
        .enumerate()
        .map(|(i, info)| {
            vk::WriteDescriptorSet::default()
                .dst_binding(i as u32)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(info)
        })
        .collect();
    let set = if pushed {
        None
    } else {
        let sizes =
            [vk::DescriptorPoolSize::default().ty(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(count as u32)];
        let pool = unsafe {
            device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&sizes), None)
        }
        .unwrap_or_else(|e| abort_with(format!("creating a descriptor pool: {e}")));
        let set_layouts = [set_layout];
        let set = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&set_layouts),
            )
        }
        .unwrap_or_else(|e| abort_with(format!("allocating a descriptor set: {e}")))[0];
        for write in &mut writes {
            write.dst_set = set;
        }
        unsafe { device.update_descriptor_sets(&writes, &[]) };
        Some((pool, set))
    };
    phase!(3);
    let written: Vec<vk::Buffer> =
        (0..count).filter(|&i| flags[i] & ARG_WRITTEN != 0).map(|i| buffers[i]).collect();
    // Every argument counted as read: the compiler's read flag is a hint.
    let command = runtime.record(&buffers, &written);
    let batch = runtime.recording.as_mut().expect("a batch");
    let query = match batch.queries {
        Some(pool) if batch.timed.len() < TIMED_PER_BATCH => {
            batch.timed.push(kernel_id);
            Some((pool, 2 * (batch.timed.len() as u32 - 1)))
        }
        _ => None,
    };
    unsafe {
        if let Some((pool, first)) = query {
            device.cmd_write_timestamp(command, vk::PipelineStageFlags::TOP_OF_PIPE, pool, first);
        }
        device.cmd_bind_pipeline(command, vk::PipelineBindPoint::COMPUTE, pipeline);
        match (set, &gpu.push_descriptor) {
            (Some((_, set)), _) => {
                device.cmd_bind_descriptor_sets(command, vk::PipelineBindPoint::COMPUTE, pipeline_layout, 0, &[set], &[])
            }
            (None, Some(push)) => {
                push.cmd_push_descriptor_set(command, vk::PipelineBindPoint::COMPUTE, pipeline_layout, 0, &writes)
            }
            (None, None) => unreachable!("a kernel pushes its buffers only with VK_KHR_push_descriptor"),
        }
        device.cmd_dispatch(command, groups_x as u32, groups_y as u32, groups_z as u32);
        if let Some((pool, first)) = query {
            device.cmd_write_timestamp(command, vk::PipelineStageFlags::BOTTOM_OF_PIPE, pool, first + 1);
        }
    }
    if let Some((pool, _)) = set {
        runtime.defer(Garbage::DescriptorPool(pool));
    }
    runtime.stats.dispatches += 1;
    phase!(4);

    for i in 0..count {
        if flags[i] & ARG_WRITTEN != 0 && copies[i].is_none() {
            runtime.mirrors.get_mut(&(pointers[i] as usize)).expect("a mirror").device_newer = true;
            bump(&DEVICE_GENERATION);
        }
    }
    // A copied argument the kernel writes comes back now; the other copies
    // live until the batch is done.
    let written_copies = (0..count).any(|i| flags[i] & ARG_WRITTEN != 0 && copies[i].is_some());
    if written_copies {
        runtime.finish();
    }
    for (i, copy) in copies.into_iter().enumerate() {
        let Some(buffer) = copy else { continue };
        if written_copies && flags[i] & ARG_WRITTEN != 0 {
            let host = unsafe { std::slice::from_raw_parts_mut(pointers[i] as usize as *mut u8, bytes[i] as usize) };
            buffer.read(host);
            runtime.stats.downloaded_bytes += bytes[i] as u64;
            runtime.profile[kernel_id].copied_out += bytes[i] as u64;
            // The host copy changed under any mirror covering it.
            for key in runtime.overlapping(pointers[i] as usize, bytes[i] as usize) {
                runtime.mirrors.get_mut(&key).expect("a mirror").host_dirty = true;
                bump(&HOST_GENERATION);
            }
        }
        runtime.defer(Garbage::Buffer(buffer));
    }
    if paranoid().1 {
        let keys: Vec<usize> = runtime.mirrors.keys().copied().collect();
        runtime.download(&keys);
    }
    if checking_nans() {
        static REPORTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        for i in 0..count {
            if flags[i] & ARG_WRITTEN == 0 {
                continue;
            }
            if flags[i] & ARG_MIRROR != 0 {
                runtime.download(&[pointers[i] as usize]);
            } else if !written_copies {
                continue;
            }
            // SAFETY: a live payload of `bytes[i]` bytes, as documented.
            let values =
                unsafe { std::slice::from_raw_parts(pointers[i] as usize as *const f32, bytes[i] as usize / 4) };
            if let Some(at) = values.iter().position(|v| !v.is_finite())
                && !REPORTED.swap(true, std::sync::atomic::Ordering::Relaxed)
            {
                eprintln!(
                    "cleave Vulkan NaN: kernel `{entry}` [{groups_x}, {groups_y}, {groups_z}] wrote {} at element {at} of argument {i} ({:#x}+{})",
                    values[at],
                    pointers[i],
                    bytes[i]
                );
                // Every argument as the kernel saw it, mirrors brought back.
                let keys: Vec<usize> =
                    (0..count).filter(|&j| flags[j] & ARG_MIRROR != 0).map(|j| pointers[j] as usize).collect();
                runtime.download(&keys);
                for j in 0..count {
                    let all = unsafe {
                        std::slice::from_raw_parts(pointers[j] as usize as *const f32, bytes[j] as usize / 4)
                    };
                    let bad = all.iter().filter(|v| !v.is_finite()).count();
                    eprintln!("  argument {j} f{}: {:?}... {bad} not finite", flags[j], &all[..all.len().min(6)]);
                }
            }
        }
    }
    if runtime.recording.as_ref().is_some_and(|b| b.commands >= BATCH_COMMANDS) {
        runtime.submit();
    }
    phase!(5);
    let _ = (phase_start, phase_waited); // the last phase's marks go unread
    runtime.stats.runtime_ns += entered.elapsed().as_nanos() as u64;
}

/// Before a kernel overwrites the whole heap payload `[ptr, ptr + bytes)`
/// without reading it (a `linalg` op writing all of its output, which isn't
/// a view): its mirror needn't be uploaded, whatever the host wrote there.
/// Inserted by the compiler (`cleave-mark-full-overwrites`).
#[unsafe(no_mangle)]
pub extern "C" fn cleave_vk_discard(ptr: i64, bytes: i64) {
    if tracing() {
        eprintln!("cleave Vulkan discard {ptr:#x}+{bytes}");
    }
    let entered = std::time::Instant::now();
    let mut guard = lock();
    let runtime = runtime(&mut guard);
    let waited = runtime.stats.wait_ns;
    runtime.mirror(ptr as usize, bytes as usize, true, &[ptr as usize]);
    runtime.stats.sync_ns += (entered.elapsed().as_nanos() as u64).saturating_sub(runtime.stats.wait_ns - waited);
}

/// Before the host reads (or partly writes) `[ptr, ptr + bytes)`: what a
/// kernel wrote there since is brought back, waiting for it. Inserted by the
/// compiler (`cleave-sync-host-accesses`); nothing to do when no mirror
/// covers it or none was written by the device.
#[unsafe(no_mangle)]
pub extern "C" fn cleave_vk_sync_host(ptr: i64, bytes: i64) {
    let call = SYNC_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    sample_caller(call, "cleave_vk_sync_host");
    if !MIRRORS_LIVE.load(std::sync::atomic::Ordering::Acquire) {
        return;
    }
    let generation = DEVICE_GENERATION.load(std::sync::atomic::Ordering::Acquire);
    if recently(&SYNCED, ptr, bytes, generation) {
        return;
    }
    if tracing() {
        eprintln!("cleave Vulkan sync_host {ptr:#x}+{bytes}");
    }
    let entered = std::time::Instant::now();
    let mut guard = lock();
    if let Some(runtime) = guard.as_mut() {
        let waited = runtime.stats.wait_ns;
        let keys = runtime.overlapping(ptr as usize, bytes as usize);
        let before = runtime.stats.downloaded_bytes;
        runtime.download(&keys);
        let downloaded = runtime.stats.downloaded_bytes - before;
        if runtime.profiling && downloaded > 0 {
            let entry = runtime.host_reads.entry(reader()).or_default();
            entry.0 += 1;
            entry.1 += downloaded;
        }
        runtime.stats.runtime_ns += entered.elapsed().as_nanos() as u64;
        runtime.stats.sync_ns += (entered.elapsed().as_nanos() as u64).saturating_sub(runtime.stats.wait_ns - waited);
        // Read under the lock: every bump happens under it.
        remember(&SYNCED, ptr, bytes, DEVICE_GENERATION.load(std::sync::atomic::Ordering::Acquire));
    }
}

/// Profiling: one host synchronization call in 4096 (the `call`-th) counted
/// against the function making it, from a backtrace: who makes millions.
fn sample_caller(call: u64, entry: &str) {
    const EVERY: u64 = 4096;
    if call % EVERY != 0 || !profiling_enabled() {
        return;
    }
    let caller = caller_of(entry);
    let mut samples = SYNC_SAMPLES.lock().unwrap_or_else(|p| p.into_inner());
    *samples.entry(format!("{entry} from {caller}")).or_default() += EVERY;
}

static SYNC_SAMPLES: Mutex<std::collections::BTreeMap<String, u64>> = Mutex::new(std::collections::BTreeMap::new());

fn profiling_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("CLEAVE_VK_PROFILE").is_ok_and(|v| v == "1"))
}

/// Profiling: the function that called [`cleave_vk_sync_host`], from a
/// backtrace (symbols permitting).
fn reader() -> String {
    caller_of("cleave_vk_sync_host")
}

/// The function that called `entry`, from a backtrace (symbols permitting).
fn caller_of(entry: &str) -> String {
    let trace = std::backtrace::Backtrace::force_capture().to_string();
    let frames: Vec<&str> = trace
        .lines()
        .filter_map(|line| {
            let (index, name) = line.trim_start().split_once(": ")?;
            index.parse::<u32>().ok().map(|_| name)
        })
        .collect();
    frames
        .iter()
        .position(|name| name.contains(entry))
        .and_then(|at| frames.get(at + 1))
        .map_or_else(|| "<unknown>".to_string(), |name| name.to_string())
}

/// After the host wrote into `[ptr, ptr + bytes)`: the mirrors covering it
/// are uploaded again before a kernel next reads them. Inserted by the
/// compiler (`cleave-sync-host-accesses`).
#[unsafe(no_mangle)]
pub extern "C" fn cleave_vk_host_wrote(ptr: i64, bytes: i64) {
    let call = WROTE_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    sample_caller(call, "cleave_vk_host_wrote");
    if !MIRRORS_LIVE.load(std::sync::atomic::Ordering::Acquire) {
        return;
    }
    let generation = HOST_GENERATION.load(std::sync::atomic::Ordering::Acquire);
    if recently(&MARKED, ptr, bytes, generation) {
        return;
    }
    if tracing() {
        eprintln!("cleave Vulkan host_wrote {ptr:#x}+{bytes}");
    }
    let entered = std::time::Instant::now();
    let mut guard = lock();
    if let Some(runtime) = guard.as_mut() {
        for key in runtime.overlapping(ptr as usize, bytes as usize) {
            runtime.mirrors.get_mut(&key).expect("a mirror").host_dirty = true;
        }
        runtime.stats.sync_ns += entered.elapsed().as_nanos() as u64;
        // Read under the lock: every bump happens under it.
        remember(&MARKED, ptr, bytes, HOST_GENERATION.load(std::sync::atomic::Ordering::Acquire));
    }
}
