module TinyKernels

export Kernel

struct Kernel{BackendType,Fun,AR,PR}
    fun::Fun
    ranges::AR
    priorities::PR
end

function Kernel(fun, ::BackendType, ranges::AR) where {BackendType,AR}
    priorities = Symbol[]
    lp_ranges = 1:1
    hp_ranges = 2:length(ranges)
    for _ in lp_ranges
        push!(priorities, :low)
    end
    for _ in hp_ranges
        push!(priorities, :high)
    end
    return Kernel{BackendType,typeof(fun),AR,typeof(priorities)}(fun, ranges, priorities)
end

include("cuda_backend.jl")

include("roc_backend.jl")

end # module TinyKernels
