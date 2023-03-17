module CPUBackend

export CPUDevice

import TinyKernels: Backend, Kernel, device_array, device_synchronize, __get_index

struct CPUDevice <: Backend end

struct CPUEvent end

import Base: wait

wait(ev::CPUEvent) = nothing
wait(evs::AbstractArray{CPUEvent}) = wait.(evs)

function split(ndrange::CartesianIndices, nthreads::NTuple)
    nblocks = ntuple(Val(length(nthreads))) do I
        return cld(size(ndrange, I), nthreads[I])
    end
    return nblocks
end

function (k::Kernel{<:CPUDevice})(args...; ndrange, priority=:low, nthreads=nothing)
    ndrange = CartesianIndices(ndrange)
    if isnothing(nthreads)
        nthreads = min(length(ndrange), 256)
    end
    nthreads = ntuple(Val(ndims(ndrange))) do I
        if I <= length(nthreads)
            nthreads[I]
        else
            1
        end
    end
    nblocks = split(ndrange, nthreads)
    off = CartesianIndex(nthreads)
    Threads.@threads for I in CartesianIndices(nblocks)
        f = first(ndrange) + CartesianIndex((Tuple(I) .- 1) .* nthreads)
        l = min(f + off - oneunit(off), last(ndrange))
        ci = f:l
        k.fun(ci, args...)
    end
    return CPUEvent()
end

device_array(::Type{T}, ::CPUDevice, dims...) where {T} = Array{T}(undef, dims)

device_synchronize(::CPUDevice) = nothing

end # module