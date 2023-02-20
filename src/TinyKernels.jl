module TinyKernels

export Kernel

struct Kernel{BackendType,Fun}
    fun::Fun
end

Kernel(fun, ::BackendType) where {BackendType} = Kernel{BackendType,typeof(fun)}(fun)

Base.similar(::Kernel{BE}, f::F) where {BE,F} = Kernel{BE,F}(f)

include("cuda_backend.jl")

include("roc_backend.jl")

include("KernelAD.jl")

end # module TinyKernels
