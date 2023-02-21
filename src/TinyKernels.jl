module TinyKernels

export Kernel, @tiny, @indices, device_array

struct Kernel{BackendType,Fun}
    fun::Fun
end

Kernel(fun, ::BackendType) where {BackendType} = Kernel{BackendType,typeof(fun)}(fun)

function __get_indices end

include("macros.jl")

@inline __validindex(ndrange) = CartesianIndex(__get_indices(Val(ndims(ndrange)))) ∈ ndrange

function device_array end

include("cuda_backend.jl")

include("roc_backend.jl")

end # module TinyKernels
