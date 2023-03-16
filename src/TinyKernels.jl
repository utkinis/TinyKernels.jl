module TinyKernels

export Kernel, @tiny, @indices, @linearindex, @cartesianindex, device_array, device_synchronize

struct Kernel{BackendType,Fun}
    fun::Fun
end

Kernel(fun, ::BackendType) where {BackendType} = Kernel{BackendType,typeof(fun)}(fun)

Base.similar(::Kernel{BE}, f::F) where {BE,F} = Kernel{BE,F}(f)

const __INDEX__ = gensym("I")

function device_array end

function device_synchronize end

function __get_index end

include("macros.jl")

include("cuda_backend.jl")

include("roc_backend.jl")

include("metal_backend.jl")

include("KernelAD.jl")

end # module TinyKernels
