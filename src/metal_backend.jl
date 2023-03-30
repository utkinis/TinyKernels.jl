module MetalBackend

export MetalDevice

import Metal

import TinyKernels: GPUDevice, Kernel, device_array, device_synchronize, __get_index, ndrange_to_indices, get_nthreads

struct MetalDevice <: GPUDevice end
struct MetalEvent
    queue::Metal.MtlCommandQueue
end

import Base: wait

wait(ev::MetalEvent) = Metal.synchronize(ev.queue)
wait(evs::AbstractArray{MetalEvent}) = wait.(evs)

mutable struct QueuePool
    next_queue_idx::Int
    queues::Vector{Metal.MtlCommandQueue}
end

const MAX_QUEUES = 6
const QUEUES = Dict{Symbol,QueuePool}()
# const QUEUES = Dict{Nothing,QueuePool}()

function get_queue(priority::Symbol) # no priority selection yet
    pool = get!(QUEUES, priority) do
        max_queues = MAX_QUEUES
        # priorities = Metal.priority_range()
        # mtl_priority = if priority == :high
        #     minimum(priorities)
        # elseif priority == :low
        #     maximum(priorities)
        # else
        #     error("unknown priority $priority")
        # end
        dev = Metal.current_device()
        QueuePool(1, [Metal.MtlCommandQueue(dev) for _ in 1:max_queues])
    end
    return pick_queue(pool)
end

function pick_queue(pool::QueuePool)
    # round-robin queue selection
    pool.next_queue_idx += 1
    pool.next_queue_idx = ((pool.next_queue_idx - 1) % length(pool.queues)) + 1
    return pool.queues[pool.next_queue_idx]
end

function (k::Kernel{<:MetalDevice})(args...; ndrange, nthreads=nothing)
    ndrange = ndrange_to_indices(ndrange)
    nthreads1 = get_nthreads(nthreads, ndrange)
    nblocks = cld(length(ndrange), nthreads1)
    # launch kernel
    queue = get_queue(:none) # no priority selection yet
    Metal.@metal threads = nthreads1 grid = nblocks k.fun(ndrange, args...)
    return MetalEvent(queue)
end

device_array(::Type{T}, ::MetalDevice, dims...) where {T} = Metal.MtlArray{T}(undef, dims)

device_synchronize(::MetalDevice) = Metal.synchronize() # device_synchronize() forces device sync

import Metal: @device_override

@device_override @inline __get_index() = Metal.thread_position_in_grid_1d()

end # module