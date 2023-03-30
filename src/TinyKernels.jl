module TinyKernels

export Kernel, GPUDevice, CPUDevice, @tiny, @indices, @linearindex, @cartesianindex, device_array, device_synchronize

struct Kernel{BackendType,Fun}
    fun::Fun
end

abstract type Backend end

abstract type GPUDevice <: Backend end

Kernel(fun, ::BackendType) where {BackendType} = Kernel{BackendType,typeof(fun)}(fun)

Base.similar(::Kernel{BE}, f::F) where {BE,F} = Kernel{BE,F}(f)

@inline ndrange_to_indices(ndrange::Tuple) = CartesianIndices(ndrange)
@inline ndrange_to_indices(ndrange::AbstractUnitRange) = CartesianIndices((ndrange,))
@inline get_nthreads(nthreads::Nothing, ndrange) = min(length(ndrange), 256)
@inline get_nthreads(nthreads, ndrange) = nthreads

const __INDEX__ = gensym("I")

function device_array end

function device_synchronize end

function __get_index end

include("macros.jl")

include("cuda_backend.jl")

include("roc_backend.jl")

include("metal_backend.jl")

include("cpu_backend.jl")

include("kernel_AD.jl")

end # module TinyKernels