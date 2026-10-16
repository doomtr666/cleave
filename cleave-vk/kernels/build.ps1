# Compiles the GPU kernels cleave-vk embeds (`include_bytes!`), with the Vulkan SDK's glslang.
$glslang = Join-Path $env:VULKAN_SDK "Bin\glslangValidator.exe"
Push-Location $PSScriptRoot
& $glslang --target-env vulkan1.3 -V gemm_bf16.comp -o gemm_bf16.spv
& $glslang --target-env vulkan1.3 -V gemm_bf16_in.comp -o gemm_bf16_in.spv
Pop-Location
