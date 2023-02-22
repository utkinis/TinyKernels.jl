module TinyKernels

export Kernel, @tiny, @indices, device_array

struct Kernel{BackendType,Fun}
    fun::Fun
end

Kernel(fun, ::BackendType) where {BackendType} = Kernel{BackendType,typeof(fun)}(fun)

Base.similar(::Kernel{BE}, f::F) where {BE,F} = Kernel{BE,F}(f)

function __get_indices end

include("macros.jl")

@inline __validindex(ndrange) = CartesianIndex(__get_indices(Val(ndims(ndrange)))) ∈ ndrange

function device_array end


include("cuda_backend.jl")

include("roc_backend.jl")

include("KernelAD.jl")

end # module TinyKernels
