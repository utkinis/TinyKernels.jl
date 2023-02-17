module TinyKernels

export Kernel

struct Kernel{BackendType,Fun}
    fun::Fun
end

Kernel(fun, ::BackendType) where {BackendType} = Kernel{BackendType,typeof(fun)}(fun)

include("cuda_backend.jl")

include("roc_backend.jl")

end # module TinyKernels
