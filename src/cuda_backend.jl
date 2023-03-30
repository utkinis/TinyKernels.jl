module CUDABackend

export CUDADevice

import CUDA

import TinyKernels: GPUDevice, Kernel, device_array, device_synchronize, __get_index, ndrange_to_indices, get_nthreads

struct CUDADevice <: GPUDevice end

struct CUDAEvent
    event::CUDA.CuEvent
end

import Base: wait

wait(ev::CUDAEvent) = CUDA.synchronize(ev.event)
wait(evs::AbstractArray{CUDAEvent}) = wait.(evs)

mutable struct StreamPool
    next_stream_idx::Int
    streams::Vector{CUDA.CuStream}
end

const MAX_STREAMS = 6
const STREAMS = Dict{Symbol,StreamPool}()

function get_stream(priority::Symbol)
    pool = get!(STREAMS, priority) do
        max_streams = MAX_STREAMS
        priorities = CUDA.priority_range()
        cu_priority = if priority == :high
            minimum(priorities)
        elseif priority == :low
            maximum(priorities)
        else
            error("unknown priority $priority")
        end
        StreamPool(1, [CUDA.CuStream(; priority=cu_priority) for _ in 1:max_streams])
    end
    return pick_stream(pool)
end

function pick_stream(pool::StreamPool)
    # round-robin stream selection
    pool.next_stream_idx += 1
    pool.next_stream_idx = ((pool.next_stream_idx - 1) % length(pool.streams)) + 1
    return pool.streams[pool.next_stream_idx]
end

function (k::Kernel{<:CUDADevice})(args...; ndrange, priority=:low, nthreads=nothing)
    ndrange = ndrange_to_indices(ndrange)
    nthreads1 = get_nthreads(nthreads, ndrange)
    nblocks = cld(length(ndrange), nthreads1)
    # generate event
    event = CUDA.CuEvent(CUDA.EVENT_DISABLE_TIMING)
    # launch kernel
    stream = get_stream(priority)
    CUDA.@cuda threads=nthreads1 blocks=nblocks stream=stream k.fun(ndrange, args...)
    # record event
    CUDA.record(event, stream)
    return CUDAEvent(event)
end

device_array(::Type{T}, ::CUDADevice, dims...) where T = CUDA.CuArray{T}(undef, dims)

device_synchronize(::CUDADevice) = CUDA.synchronize()

import CUDA: @device_override

@device_override @inline __get_index() = (CUDA.blockIdx().x-1)*CUDA.blockDim().x + CUDA.threadIdx().x

end # module