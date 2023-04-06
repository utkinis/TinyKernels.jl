module AMDGPUExt

@static if isdefined(Base, :get_extension)
    import AMDGPU
    import AMDGPU: @device_override
else
    import ..AMDGPU
    import ..AMDGPU: @device_override
end

import TinyKernels: AMDGPUDevice, AbstractEvent, Kernel
import TinyKernels: device_array, device_synchronize, __get_index,  ndrange_to_indices

import Base: wait

struct AMDGPUEvent <: AbstractEvent
    signal::AMDGPU.ROCSignal
    queue::AMDGPU.ROCQueue
end

wait(ev::AMDGPUEvent) = wait(ev.signal; queue=ev.queue)
wait(evs::AbstractArray{AMDGPUEvent}) = wait.(evs)

mutable struct QueuePool
    next_queue_idx::Int
    queues::Vector{AMDGPU.ROCQueue}
end

const MAX_QUEUES = 2
const QUEUES = Dict{Symbol,QueuePool}()

function get_queue(priority::Symbol)
    pool = get!(QUEUES, priority) do
        max_queues = MAX_QUEUES
        roc_priority = if priority == :high
            :high
        elseif priority == :low
            :low
        else
            error("unknown priority $priority")
        end
        QueuePool(1, [AMDGPU.ROCQueue(AMDGPU.default_device(); priority=roc_priority) for _ in 1:max_queues])
    end
    return pick_queue(pool)
end

function pick_queue(pool::QueuePool)
    # round-robin queue selection
    pool.next_queue_idx += 1
    pool.next_queue_idx = ((pool.next_queue_idx - 1) % length(pool.queues)) + 1
    return pool.queues[pool.next_queue_idx]
end

function (k::Kernel{<:AMDGPUDevice})(args...; ndrange, priority=:low, nthreads=nothing)
    ndrange = ndrange_to_indices(ndrange)
    if isnothing(nthreads)
        nthreads = min(length(ndrange), 256)
    end
    ngrid = length(ndrange)
    # create signal
    sig = AMDGPU.ROCSignal()
    # launch kernel
    queue = get_queue(priority)
    AMDGPU.HSA.signal_store_screlease(sig.signal, 1)
    AMDGPU.@roc wait=false mark=false signal=sig groupsize=nthreads gridsize=ngrid queue=queue k.fun(ndrange, args...)
    return AMDGPUEvent(sig, queue)
end

device_array(::Type{T}, ::AMDGPUDevice, dims...) where T = AMDGPU.ROCArray{T}(undef, dims)

function device_synchronize(::AMDGPUDevice)
    wait(AMDGPU.barrier_and!(AMDGPU.default_queue(), AMDGPU.active_kernels(AMDGPU.default_queue())))
    return
end

@device_override @inline __get_index() = (AMDGPU.workgroupIdx().x-1)*AMDGPU.workgroupDim().x + AMDGPU.workitemIdx().x

end