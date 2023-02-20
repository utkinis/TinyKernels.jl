module CUDABackend

export CUDADevice

using CUDA

import TinyKernels: Kernel

struct CUDADevice end

struct CUDAEvent
    event::CuEvent
end

import Base: wait

wait(ev::CUDAEvent) = synchronize(ev.event)
wait(evs::AbstractArray{CUDAEvent}) = wait.(evs)

mutable struct StreamPool
    next_stream_idx::Int
    streams::Vector{CuStream}
end

const MAX_STREAMS = 6
const STREAMS = Dict{Symbol,StreamPool}()

function get_stream(priority::Symbol)
    pool = get!(STREAMS, priority) do
        max_streams = MAX_STREAMS
        priorities = priority_range()
        cu_priority = if priority == :high
            minimum(priorities)
        elseif priority == :low
            maximum(priorities)
        else
            error("unknown priority $priority")
        end
        StreamPool(1, [CuStream(; priority=cu_priority) for _ in 1:max_streams])
    end
    return pick_stream(pool)
end

function pick_stream(pool::StreamPool)
    # round-robin stream selection
    pool.next_stream_idx += 1
    pool.next_stream_idx = ((pool.next_stream_idx - 1) % length(pool.streams)) + 1
    return pool.streams[pool.next_stream_idx]
end

function (k::Kernel{<:CUDADevice})(args...; range, priority=:low)
    # compile CUDA kernel
    cu_kernel = @cuda launch=false k.fun(range, args...)
    # determine optimal launch parameters
    config = CUDA.launch_configuration(cu_kernel.fun)
    nthreads = (32, cld(config.threads, 32))
    nblocks = cld.(length.(range), nthreads)
    # generate event
    event = CuEvent(CUDA.EVENT_DISABLE_TIMING)
    # launch kernel
    stream = get_stream(priority)
    cu_kernel(range, args...; threads=nthreads, blocks=nblocks, stream=stream)
    # record event
    CUDA.record(event, stream)
    return CUDAEvent(event)
end

end
