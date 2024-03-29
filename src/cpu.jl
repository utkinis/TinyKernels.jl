module CPU

import TinyKernels: Kernel, AbstractEvent, CPUDevice
import TinyKernels: device_array, device_synchronize, __get_index, ndrange_to_indices, get_nthreads

import Base: wait

struct CPUEvent <: AbstractEvent end

wait(ev::CPUEvent) = nothing
wait(evs::AbstractArray{CPUEvent}) = wait.(evs)

@inline function split(ndrange::CartesianIndices, nthreads::NTuple)
    nblocks = ntuple(Val(length(nthreads))) do I
        return cld(size(ndrange, I), nthreads[I])
    end
    return nblocks
end

function (k::Kernel{<:CPUDevice})(args...; ndrange, priority=:low, nthreads=nothing)
    ndrange = ndrange_to_indices(ndrange)
    nthreads1 = get_nthreads(nthreads, ndrange)
    n = length(nthreads1)
    nthreads2 = ntuple(Val(ndims(ndrange))) do I
        Base.@_inline_meta
        I ≤ n ? nthreads1[I] : 1
    end
    nblocks = split(ndrange, nthreads2)
    off = CartesianIndex(nthreads2)
    Threads.@threads for I in CartesianIndices(nblocks)
        f = first(ndrange) + CartesianIndex((Tuple(I) .- 1) .* nthreads2)
        l = min(f + off - oneunit(off), last(ndrange))
        ci = f:l
        k.fun(ci, args...)
    end
    return CPUEvent()
end

device_array(::Type{T}, ::CPUDevice, dims...) where {T} = Array{T}(undef, dims)

device_synchronize(::CPUDevice) = nothing

end # module