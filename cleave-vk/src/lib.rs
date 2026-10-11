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
}

/// Every Vulkan device the loader finds, in its order.
pub fn devices() -> Result<Vec<DeviceInfo>, Error> {
    let (entry, instance) = create_instance(false)?;
    let found = unsafe { instance.enumerate_physical_devices() }
        .map_err(fail("enumerating devices"))
        .map(|physicals| physicals.iter().map(|&p| device_info(&instance, p)).collect());
    unsafe { instance.destroy_instance(None) };
    drop(entry);
    found
}

fn device_info(instance: &ash::Instance, physical: vk::PhysicalDevice) -> DeviceInfo {
    let properties = unsafe { instance.get_physical_device_properties(physical) };
    let name = properties.device_name_as_c_str().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    DeviceInfo { name, discrete: properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU }
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
    /// Nanoseconds per timestamp tick (`CLEAVE_VK_PROFILE`).
    timestamp_period: f32,
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
        let infos: Vec<DeviceInfo> = physicals.iter().map(|&p| device_info(&instance, p)).collect();
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
        let device = unsafe { instance.create_device(physical, &vk::DeviceCreateInfo::default().queue_create_infos(&queues), None) }
            .map_err(fail("creating the device"))?;
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
        let allocator = unsafe { vk_mem::Allocator::new(vk_mem::AllocatorCreateInfo::new(&instance, &device, physical)) }
            .map_err(fail("creating the memory allocator"))?;

        let timestamp_period = unsafe { instance.get_physical_device_properties(physical) }.limits.timestamp_period;
        Ok(Gpu {
            info: infos[chosen].clone(),
            allocator: std::mem::ManuallyDrop::new(allocator),
            device,
            queue,
            command_pool,
            timestamp_period,
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
        let allocation_info = vk_mem::AllocationCreateInfo { usage, flags: access | mapped_flag, ..Default::default() };
        let (buffer, allocation) =
            unsafe { self.allocator.create_buffer(&info, &allocation_info) }.map_err(fail("allocating a buffer"))?;
        let mapped = self.allocator.get_allocation_info(&allocation).mapped_data as *mut u8;
        Ok(Buffer { gpu: self, buffer, allocation, bytes, mapped })
    }

    /// A compute kernel: the entry point `entry` of the SPIR-V module
    /// `spirv`, reading and writing `bindings` storage buffers (set 0,
    /// bindings `0..bindings`, in the order a dispatch passes them).
    pub fn kernel(&self, spirv: &[u32], entry: &str, bindings: u32) -> Result<Kernel<'_>, Error> {
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
        let set_layout = unsafe {
            device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&layout_bindings), None)
        }
        .map_err(fail("creating the descriptor set layout"))?;
        let set_layouts = [set_layout];
        let pipeline_layout =
            unsafe { device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts), None) }
                .map_err(fail("creating the pipeline layout"))?;
        let name = CString::new(entry).map_err(|_| Error(format!("entry point name {entry:?}")))?;
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(&name);
        let pipeline = unsafe {
            device.create_compute_pipelines(
                vk::PipelineCache::null(),
                &[vk::ComputePipelineCreateInfo::default().stage(stage).layout(pipeline_layout)],
                None,
            )
        }
        .map_err(|(_, result)| fail("creating the compute pipeline")(result))?[0];
        unsafe { device.destroy_shader_module(module, None) };
        Ok(Kernel { gpu: self, bindings, set_layout, pipeline_layout, pipeline })
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
            let set_layouts = [kernel.set_layout];
            let set = unsafe {
                device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&set_layouts))
            }
            .map_err(fail("allocating a descriptor set"))?[0];
            let infos: Vec<[vk::DescriptorBufferInfo; 1]> = buffers
                .iter()
                .map(|b| [vk::DescriptorBufferInfo::default().buffer(b.buffer).offset(0).range(vk::WHOLE_SIZE)])
                .collect();
            let writes: Vec<vk::WriteDescriptorSet> = infos
                .iter()
                .enumerate()
                .map(|(i, info)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(i as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(info)
                })
                .collect();
            unsafe { device.update_descriptor_sets(&writes, &[]) };

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
                device.cmd_bind_descriptor_sets(command, vk::PipelineBindPoint::COMPUTE, kernel.pipeline_layout, 0, &[set], &[]);
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
}

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
}

/// What a batch keeps alive until the device has finished it: what its
/// commands use and nothing else holds any more.
enum Garbage {
    Buffer(Buffer<'static>),
    DescriptorPool(vk::DescriptorPool),
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
}

/// A batch is submitted, without waiting for it, once it holds this many
/// commands: the device starts on long sequences before the host needs
/// anything back, and what a batch keeps alive stays bounded.
const BATCH_COMMANDS: usize = 256;

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
}

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
    let _ = writeln!(
        out,
        "  host synchronizations: {} sync_host, {} host_wrote calls",
        SYNC_CALLS.load(std::sync::atomic::Ordering::Relaxed),
        WROTE_CALLS.load(std::sync::atomic::Ordering::Relaxed)
    );
    let total: f64 = runtime.profile.iter().map(|k| k.gpu_ns).sum();
    let _ = writeln!(out, "  device time in kernels: {:.1} ms", total / 1e6);
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
    for k in kernels.iter().take(30) {
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
    /// The command buffer being recorded, begun if none is, with a full
    /// memory barrier recorded after what it already holds: every command
    /// sees every earlier one's writes, transfers and dispatches alike.
    fn record(&mut self) -> vk::CommandBuffer {
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
            self.recording = Some(Batch { command, fence, commands: 0, garbage: Vec::new(), queries, timed: Vec::new() });
        }
        let batch = self.recording.as_mut().expect("a batch");
        // Before every command, the first of a batch too: a barrier's first
        // scope is everything submitted to the queue before it, and an earlier
        // batch may still be running (submitted without waiting).
        {
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
        batch.commands += 1;
        batch.command
    }

    /// Keeps `garbage` alive until the device has finished everything that
    /// may use it: the batch being recorded, else the last one in flight;
    /// dropped at once when nothing is pending.
    fn defer(&mut self, garbage: Garbage) {
        if let Some(batch) = self.recording.as_mut().or(self.in_flight.last_mut()) {
            batch.garbage.push(garbage);
        } else {
            self.dispose(garbage);
        }
    }

    fn dispose(&self, garbage: Garbage) {
        match garbage {
            Garbage::Buffer(buffer) => drop(buffer),
            Garbage::DescriptorPool(pool) => unsafe { self.gpu.device.destroy_descriptor_pool(pool, None) },
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
        for message in self.gpu.validation_messages() {
            eprintln!("cleave Vulkan validation: {message}");
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
            let staging = gpu.buffer(len).unwrap_or_else(|e| abort_with(format!("staging {len} bytes: {e}")));
            let command = self.record();
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
            staging.read(host);
            mirror.device_newer = false;
            self.stats.downloaded_bytes += mirror.len as u64;
            if tracing() {
                eprintln!("cleave Vulkan download {key:#x}+{}", mirror.len);
            }
        }
    }

    /// Records an upload of the host's `[start, start + len)` into `target`:
    /// the bytes copied now into a fresh staging buffer, which nothing in
    /// flight uses, and copied on the device in order with what's recorded.
    fn upload(&mut self, start: usize, len: usize, target: vk::Buffer) {
        let gpu = self.gpu;
        let mut staging = gpu.buffer(len).unwrap_or_else(|e| abort_with(format!("staging {len} bytes: {e}")));
        // SAFETY: the caller's range is live host memory.
        staging.write(unsafe { std::slice::from_raw_parts(start as *const u8, len) });
        let command = self.record();
        let region = [vk::BufferCopy::default().size(len as u64)];
        unsafe { gpu.device.cmd_copy_buffer(command, staging.buffer, target, &region) };
        self.defer(Garbage::Buffer(staging));
        self.stats.uploaded_bytes += len as u64;
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
    fn mirror(&mut self, start: usize, len: usize, discard: bool) {
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
        let gpu = self.gpu;
        if !self.mirrors.contains_key(&start) {
            let buffer = gpu.device_buffer(len).unwrap_or_else(|e| abort_with(format!("mirror of {len} bytes: {e}")));
            self.mirrors.insert(start, Mirror { len, buffer, host_dirty: true, device_newer: false });
            MIRRORS_LIVE.store(true, std::sync::atomic::Ordering::Release);
            bump(&DEVICE_GENERATION);
            bump(&HOST_GENERATION);
        }
        let mirror = self.mirrors.get_mut(&start).expect("a mirror");
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
    let entered = std::time::Instant::now();
    let mut guard = lock();
    let runtime = runtime(&mut guard);
    let gpu = runtime.gpu;
    // SAFETY: as documented, every pointer is valid for what it describes.
    let entry = unsafe { CStr::from_ptr(entry as usize as *const std::ffi::c_char) }.to_string_lossy().into_owned();
    let count = count as usize;
    let pointers = unsafe { std::slice::from_raw_parts(pointers as usize as *const i64, count) };
    let bytes = unsafe { std::slice::from_raw_parts(bytes as usize as *const i64, count) };
    let flags = unsafe { std::slice::from_raw_parts(flags as usize as *const i64, count) };
    if tracing() {
        let args: Vec<String> =
            (0..count).map(|i| format!("{:#x}+{} f{}", pointers[i], bytes[i], flags[i])).collect();
        eprintln!("cleave Vulkan dispatch `{entry}` [{groups_x}, {groups_y}, {groups_z}]: {}", args.join(", "));
    }
    let key = (spirv as usize, entry.clone());
    if !runtime.kernels.contains_key(&key) {
        let binary = unsafe { std::slice::from_raw_parts(spirv as usize as *const u32, words as usize) };
        let kernel =
            gpu.kernel(binary, &entry, count as u32).unwrap_or_else(|e| abort_with(format!("kernel `{entry}`: {e}")));
        runtime.kernels.insert(key.clone(), kernel);
        let id = runtime.profile.len();
        runtime.profile.push(KernelProfile {
            name: format!("{entry} #{id} [{groups_x}, {groups_y}, {groups_z}]"),
            ..Default::default()
        });
        runtime.kernel_ids.insert(key.clone(), id);
    }
    let kernel_id = runtime.kernel_ids[&key];
    runtime.profile[kernel_id].dispatches += 1;
    let (uploaded_before, downloaded_before) = (runtime.stats.uploaded_bytes, runtime.stats.downloaded_bytes);

    if paranoid().0 {
        // Not over a newer device copy: that one is the truth.
        for mirror in runtime.mirrors.values_mut().filter(|m| !m.device_newer) {
            mirror.host_dirty = true;
        }
    }
    // The mirrors first (each may retire another), then the copies.
    for i in 0..count {
        if flags[i] & ARG_MIRROR != 0 {
            runtime.mirror(pointers[i] as usize, bytes[i] as usize, false);
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
    let mut copies: Vec<Option<Buffer<'static>>> = Vec::with_capacity(count);
    for i in 0..count {
        if flags[i] & ARG_MIRROR != 0 {
            copies.push(None);
            continue;
        }
        let mut buffer =
            gpu.buffer(bytes[i] as usize).unwrap_or_else(|e| abort_with(format!("kernel `{entry}`: {e}")));
        // Copied in whether the kernel reads it or not: one that writes only
        // part of it would bring the rest back as garbage otherwise.
        let host = unsafe { std::slice::from_raw_parts(pointers[i] as usize as *const u8, bytes[i] as usize) };
        buffer.write(host);
        runtime.stats.uploaded_bytes += bytes[i] as u64;
        runtime.profile[kernel_id].copied_in += bytes[i] as u64;
        copies.push(Some(buffer));
    }
    let buffers: Vec<vk::Buffer> = (0..count)
        .map(|i| match &copies[i] {
            Some(buffer) => buffer.buffer,
            None => runtime.mirrors[&(pointers[i] as usize)].buffer.buffer,
        })
        .collect();

    // The descriptor set, from a pool of its own, kept until the batch is done.
    let kernel = &runtime.kernels[&key];
    let (pipeline, pipeline_layout, set_layout) = (kernel.pipeline, kernel.pipeline_layout, kernel.set_layout);
    let device = &gpu.device;
    let sizes = [vk::DescriptorPoolSize::default().ty(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(count as u32)];
    let pool =
        unsafe { device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&sizes), None) }
            .unwrap_or_else(|e| abort_with(format!("creating a descriptor pool: {e}")));
    let set_layouts = [set_layout];
    let set = unsafe {
        device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&set_layouts))
    }
    .unwrap_or_else(|e| abort_with(format!("allocating a descriptor set: {e}")))[0];
    let infos: Vec<[vk::DescriptorBufferInfo; 1]> = buffers
        .iter()
        .map(|&b| [vk::DescriptorBufferInfo::default().buffer(b).offset(0).range(vk::WHOLE_SIZE)])
        .collect();
    let writes: Vec<vk::WriteDescriptorSet> = infos
        .iter()
        .enumerate()
        .map(|(i, info)| {
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(i as u32)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(info)
        })
        .collect();
    unsafe { device.update_descriptor_sets(&writes, &[]) };
    let command = runtime.record();
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
        device.cmd_bind_descriptor_sets(command, vk::PipelineBindPoint::COMPUTE, pipeline_layout, 0, &[set], &[]);
        device.cmd_dispatch(command, groups_x as u32, groups_y as u32, groups_z as u32);
        if let Some((pool, first)) = query {
            device.cmd_write_timestamp(command, vk::PipelineStageFlags::BOTTOM_OF_PIPE, pool, first + 1);
        }
    }
    runtime.defer(Garbage::DescriptorPool(pool));
    runtime.stats.dispatches += 1;

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
    let mut guard = lock();
    let runtime = runtime(&mut guard);
    runtime.mirror(ptr as usize, bytes as usize, true);
}

/// Before the host reads (or partly writes) `[ptr, ptr + bytes)`: what a
/// kernel wrote there since is brought back, waiting for it. Inserted by the
/// compiler (`cleave-sync-host-accesses`); nothing to do when no mirror
/// covers it or none was written by the device.
#[unsafe(no_mangle)]
pub extern "C" fn cleave_vk_sync_host(ptr: i64, bytes: i64) {
    SYNC_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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
        // Read under the lock: every bump happens under it.
        remember(&SYNCED, ptr, bytes, DEVICE_GENERATION.load(std::sync::atomic::Ordering::Acquire));
    }
}

/// Profiling: the function that called [`cleave_vk_sync_host`], from a
/// backtrace (symbols permitting).
fn reader() -> String {
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
        .position(|name| name.contains("cleave_vk_sync_host"))
        .and_then(|at| frames.get(at + 1))
        .map_or_else(|| "<unknown>".to_string(), |name| name.to_string())
}

/// After the host wrote into `[ptr, ptr + bytes)`: the mirrors covering it
/// are uploaded again before a kernel next reads them. Inserted by the
/// compiler (`cleave-sync-host-accesses`).
#[unsafe(no_mangle)]
pub extern "C" fn cleave_vk_host_wrote(ptr: i64, bytes: i64) {
    WROTE_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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
    let mut guard = lock();
    if let Some(runtime) = guard.as_mut() {
        for key in runtime.overlapping(ptr as usize, bytes as usize) {
            runtime.mirrors.get_mut(&key).expect("a mirror").host_dirty = true;
        }
        // Read under the lock: every bump happens under it.
        remember(&MARKED, ptr, bytes, HOST_GENERATION.load(std::sync::atomic::Ordering::Acquire));
    }
}
