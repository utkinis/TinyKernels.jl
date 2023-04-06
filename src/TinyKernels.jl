module TinyKernels

export Kernel, AbstractDevice, AbstractGPUDevice
export CPUDevice, CUDADevice, AMDGPUDevice, MetalDevice
export device_array, device_synchronize, @tiny, @indices, @linearindex, @cartesianindex

if !isdefined(Base, :get_extension)
    using Requires
end

struct Kernel{DeviceType,Fun}
    fun::Fun
end

abstract type AbstractDevice end

abstract type AbstractGPUDevice <: AbstractDevice end

abstract type AbstractEvent end

struct CPUDevice <: AbstractDevice end

struct CUDADevice <: AbstractGPUDevice end

struct AMDGPUDevice <: AbstractGPUDevice end

struct MetalDevice <: AbstractGPUDevice end

Kernel(fun, ::DeviceType) where {DeviceType} = Kernel{DeviceType,typeof(fun)}(fun)

function (k::Kernel{<:AbstractDevice})(args...; ndrange, priority=:low, nthreads=nothing)
    @warn "no device API loaded, skipping"
    return
end

Base.similar(::Kernel{D}, f::F) where {D,F} = Kernel{D,F}(f)

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

include("cpu.jl")

@static if !isdefined(Base, :get_extension)
    function __init__()
        @require AMDGPU="21141c5a-9bdb-4563-92ae-f87d6854732e" include("../ext/AMDGPUExt.jl")
        @require CUDA="052768ef-5323-5732-b1bb-66c8b64840ba" include("../ext/CUDAExt.jl")
        @require Metal="dde4c033-4e86-420c-a63e-0dd931031962" include("../ext/MetalExt.jl")
        @require Enzyme="7da242da-08ed-463a-9acd-ee780be4f1d9" include("../ext/EnzymeExt.jl")
    end
end

end # module TinyKernels