module AMDGPUExt

@static if isdefined(Base, :get_extension)
    import AMDGPU
    import AMDGPU.Device: @device_override
else
    import ..AMDGPU
    import ..AMDGPU.Device: @device_override
end

import TinyKernels: AMDGPUDevice, AbstractEvent, Kernel
import TinyKernels: device_array, device_synchronize, synchronize, __get_index,  ndrange_to_indices

import Base: wait

struct AMDGPUEvent <: AbstractEvent
    event::AMDGPU.HIP.HIPEvent
end

wait(ev::AMDGPUEvent) = AMDGPU.HIP.synchronize(ev.event)
wait(evs::AbstractArray{AMDGPUEvent}) = wait.(evs)

mutable struct StreamPool
    next_stream_idx::Int
    streams::Vector{AMDGPU.HIPStream}
end

const MAX_STREAMS = 6
const STREAMS = Dict{Symbol,StreamPool}()

function get_stream(priority::Symbol)
    pool = get!(STREAMS, priority) do
        max_streams = MAX_STREAMS
        roc_priority = if priority == :high
            :high
        elseif priority == :low
            :low
        else
            error("unknown priority $priority")
        end
        StreamPool(0, [AMDGPU.HIPStream(roc_priority) for _ in 1:max_streams])
    end
    return pick_stream(pool)
end

function pick_stream(pool::StreamPool)
    # round-robin stream selection
    pool.next_stream_idx += 1
    pool.next_stream_idx = ((pool.next_stream_idx - 1) % length(pool.streams)) + 1
    return pool.streams[pool.next_stream_idx]
end

function (k::Kernel{<:AMDGPUDevice})(args...; ndrange, priority=:low, nthreads=nothing)
    ndrange = ndrange_to_indices(ndrange)
    if isnothing(nthreads)
        nthreads = min(length(ndrange), 256)
    end
    nblocks = cld(length(ndrange), nthreads)
    stream = get_stream(priority)
    event = AMDGPU.HIP.HIPEvent(stream)
    AMDGPU.@roc groupsize=nthreads gridsize=nblocks stream=stream k.fun(ndrange, args...)
    AMDGPU.HIP.record(event)
    return AMDGPUEvent(event)
end

device_array(::Type{T}, ::AMDGPUDevice, dims...) where T = AMDGPU.ROCArray{T}(undef, dims)

device_synchronize(::AMDGPUDevice) = AMDGPU.HIP.device_synchronize()

synchronize(::AMDGPUDevice) = AMDGPU.synchronize()

@device_override @inline __get_index() = (AMDGPU.workgroupIdx().x-1)*AMDGPU.workgroupDim().x + AMDGPU.workitemIdx().x

end # module