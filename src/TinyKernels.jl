module TinyKernels

export Kernel, GPUDevice, CPUDevice, @tiny, @indices, @linearindex, @cartesianindex, device_array, device_synchronize

if !isdefined(Base, :get_extension)
    using Requires
end

struct Kernel{BackendType,Fun}
    fun::Fun
end

abstract type Backend end

abstract type GPUDevice <: Backend end

Kernel(fun, ::BackendType) where {BackendType} = Kernel{BackendType,typeof(fun)}(fun)

Base.similar(::Kernel{BE}, f::F) where {BE,F} = Kernel{BE,F}(f)

const __INDEX__ = gensym("I")

function device_array end

function device_synchronize end

function __get_index end

include("macros.jl")

include("CPUBackend.jl")

include("KernelAD.jl")

@static if !isdefined(Base, :get_extension)
    @require CUDA="052768ef-5323-5732-b1bb-66c8b64840ba" include("../backends/CUDABackend.jl")
    @require AMDGPU="21141c5a-9bdb-4563-92ae-f87d6854732e" include("../backends/ROCBackend.jl")
    @require Metal="dde4c033-4e86-420c-a63e-0dd931031962" include("../backends/MetalBackend.jl")
end

end # module TinyKernels