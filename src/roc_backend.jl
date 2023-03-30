module ROCBackend

export ROCDevice

import AMDGPU

import TinyKernels: GPUDevice, Kernel, __get_index, device_array, device_synchronize, ndrange_to_indices, get_nthreads

struct ROCDevice <: GPUDevice end

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

function (k::Kernel{<:ROCDevice})(args...; ndrange, priority=:low, nthreads=nothing)
    ndrange = ndrange_to_indices(ndrange)
    nthreads1 = get_nthreads(nthreads, ndrange)
    ngrid = length(ndrange)
    # create signal
    sig = AMDGPU.ROCSignal()
    # launch kernel
    queue = get_queue(priority)
    AMDGPU.HSA.signal_store_screlease(sig.signal, 1)
    AMDGPU.@roc wait=false mark=false signal=sig groupsize=nthreads1 gridsize=ngrid queue=queue k.fun(ndrange, args...)
    return ROCEvent(sig, queue)
end

device_array(::Type{T}, ::ROCDevice, dims...) where T = AMDGPU.ROCArray{T}(undef, dims)

function device_synchronize(::ROCDevice)
    wait(AMDGPU.barrier_and!(AMDGPU.default_queue(), AMDGPU.active_kernels(AMDGPU.default_queue())))
    return
end

import AMDGPU.Device: @device_override

@device_override @inline __get_index() = (AMDGPU.workgroupIdx().x-1)*AMDGPU.workgroupDim().x + AMDGPU.workitemIdx().x

end