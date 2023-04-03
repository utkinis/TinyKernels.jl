module TinyKernels

export Kernel, GPUDevice, CPUDevice, device_array, device_synchronize, @tiny, @indices, @linearindex, @cartesianindex

if !isdefined(Base, :get_extension)
    using Requires
end

struct Kernel{BackendType,Fun}
    fun::Fun
end

abstract type Backend end

abstract type GPUDevice <: Backend end

struct CPUDevice <: Backend end

Kernel(fun, ::BackendType) where {BackendType} = Kernel{BackendType,typeof(fun)}(fun)

Base.similar(::Kernel{BE}, f::F) where {BE,F} = Kernel{BE,F}(f)

@inline ndrange_to_indices(ndrange::Tuple) = CartesianIndices(ndrange)
@inline ndrange_to_indices(ndrange::AbstractUnitRange) = CartesianIndices((ndrange,))
@inline ndrange_to_indices(ndrange::CartesianIndices)  = ndrange

@inline get_nthreads(nthreads::Nothing, ndrange) = min(length(ndrange), 256)
@inline get_nthreads(nthreads, ndrange) = nthreads

const __INDEX__ = gensym("I")

function device_array end

function device_synchronize end

function __get_index end

include("macros.jl")

include("CPUBackend.jl")

@static if !isdefined(Base, :get_extension)
    function __init__()
        @require AMDGPU="21141c5a-9bdb-4563-92ae-f87d6854732e" include("../ext/ROCBackend.jl")
        @require CUDA="052768ef-5323-5732-b1bb-66c8b64840ba" include("../ext/CUDABackend.jl")
        @require Enzyme="7da242da-08ed-463a-9acd-ee780be4f1d9" include("../ext/KernelAD.jl")
        @require Metal="dde4c033-4e86-420c-a63e-0dd931031962" include("../ext/MetalBackend.jl")
    end
end

end # module TinyKernels