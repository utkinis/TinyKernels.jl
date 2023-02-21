module ROCBackend

export ROCDevice

import TinyKernels: Kernel, __get_indices, device_array

import AMDGPU

struct ROCDevice end

struct ROCEvent
    signal::AMDGPU.ROCSignal
    queue::AMDGPU.ROCQueue
end

import Base: wait

wait(ev::ROCEvent) = wait(ev.signal; queue=ev.queue)
wait(evs::AbstractArray{ROCEvent}) = wait.(evs)

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

function (k::Kernel{<:ROCDevice})(args...; range, priority=:low)
    ndrange = CartesianIndices(range)
    # compile ROC kernel
    roc_kernel = AMDGPU.@roc launch=false k.fun(ndrange, args...)
    # determine optimal launch parameters
    config = AMDGPU.launch_configuration(roc_kernel.fun)
    nthreads = ntuple(length(range)) do i
        if i == 1
            min(range[1], 32)
        elseif i == 2
            min(range[2], cld(config.groupsize, 32))
        elseif i == 3
            min(range[3], 1)
        end
    end
    # create signal
    sig = AMDGPU.ROCSignal()
    # launch kernel
    queue = get_queue(priority)
    AMDGPU.HSA.signal_store_screlease(sig.signal, 1)
    AMDGPU.@roc wait=false mark=false signal=sig groupsize=nthreads gridsize=range queue=queue k.fun(ndrange, args...)
    return ROCEvent(sig, queue)
end

device_array(::Type{T}, ::ROCDevice, dims...) where T = AMDGPU.ROCArray{T}(undef, dims)

import AMDGPU.Device: @device_override

@device_override @inline __get_indices(::Val{1}) = (AMDGPU.workgroupIdx().x-1)*AMDGPU.workgroupDim().x + AMDGPU.workitemIdx().x

@device_override @inline function __get_indices(::Val{2})
    ix = (AMDGPU.workgroupIdx().x-1)*AMDGPU.workgroupDim().x + AMDGPU.workitemIdx().x
    iy = (AMDGPU.workgroupIdx().y-1)*AMDGPU.workgroupDim().y + AMDGPU.workitemIdx().y
    return ix, iy
end

@device_override @inline function __get_indices(::Val{3})
    ix = (AMDGPU.workgroupIdx().x-1)*AMDGPU.workgroupDim().x + AMDGPU.workitemIdx().x
    iy = (AMDGPU.workgroupIdx().y-1)*AMDGPU.workgroupDim().y + AMDGPU.workitemIdx().y
    iz = (AMDGPU.workgroupIdx().z-1)*AMDGPU.workgroupDim().z + AMDGPU.workitemIdx().z
    return ix, iy, iz
end

end
