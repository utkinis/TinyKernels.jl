module MetalExt

@static if isdefined(Base, :get_extension)
    import Metal
    import Metal: @device_override
else
    import ..Metal
    import ..Metal: @device_override
end

import TinyKernels: MetalDevice, AbstractEvent, Kernel
import TinyKernels: device_array, device_synchronize, __get_index, ndrange_to_indices

import Base: wait

struct MetalEvent <: AbstractEvent
    queue::Metal.MTLCommandQueue
end

wait(ev::MetalEvent) = Metal.synchronize(ev.queue)
wait(evs::AbstractArray{MetalEvent}) = wait.(evs)

mutable struct QueuePool
    next_queue_idx::Int
    queues::Vector{Metal.MTLCommandQueue}
end

const MAX_QUEUES = 6
const QUEUES = Dict{Symbol,QueuePool}()

function get_queue(priority::Symbol) # no priority selection yet
    pool = get!(QUEUES, priority) do
        max_queues = MAX_QUEUES
        dev = Metal.current_device()
        QueuePool(0, [Metal.MTLCommandQueue(dev) for _ in 1:max_queues])
    end
    return pick_queue(pool)
end

function pick_queue(pool::QueuePool)
    # round-robin queue selection
    pool.next_queue_idx += 1
    pool.next_queue_idx = ((pool.next_queue_idx - 1) % length(pool.queues)) + 1
    return pool.queues[pool.next_queue_idx]
end

function (k::Kernel{<:MetalDevice})(args...; ndrange, priority=:low, nthreads=nothing)
    ndrange = ndrange_to_indices(ndrange)
    if isnothing(nthreads)
        nthreads = min(length(ndrange), 256)
    end
    nblocks = cld(length(ndrange), nthreads)
    # launch kernel
    queue = get_queue(priority) # no priority selection yet
    Metal.@metal threads = nthreads groups = nblocks k.fun(ndrange, args...)
    return MetalEvent(queue)
end

device_array(::Type{T}, ::MetalDevice, dims...) where {T} = Metal.MtlArray{T}(undef, dims)

device_synchronize(::MetalDevice) = Metal.synchronize() # device_synchronize() forces device sync

@device_override @inline __get_index() = Metal.thread_position_in_grid_1d()

end # module