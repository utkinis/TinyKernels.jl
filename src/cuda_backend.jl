module CUDABackend

export CUDADevice

import CUDA

import TinyKernels: Kernel, __get_indices, device_array

struct CUDADevice end

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

function (k::Kernel{<:CUDADevice})(args...; range, priority=:low)
    ndrange = CartesianIndices(range)
    # compile CUDA kernel
    cu_kernel = CUDA.@cuda launch = false k.fun(ndrange, args...)
    # determine optimal launch parameters
    config = CUDA.launch_configuration(cu_kernel.fun)
    nthreads = ntuple(length(range)) do i
        if i == 1
            min(range[1], 32)
        elseif i == 2
            min(range[2], cld(config.threads, 32))
        elseif i == 3
            min(range[3], 1)
        end
    end
    nblocks = cld.(range, nthreads)
    # generate event
    event = CUDA.CuEvent(CUDA.EVENT_DISABLE_TIMING)
    # launch kernel
    stream = get_stream(priority)
    cu_kernel(ndrange, args...; threads=nthreads, blocks=nblocks, stream=stream)
    # record event
    CUDA.record(event, stream)
    return CUDAEvent(event)
end

device_array(::Type{T}, ::CUDADevice, dims...) where T = CuArray{T}(undef, dims)

import CUDA: @device_override

@device_override @inline __get_indices(::Val{1}) = (CUDA.blockIdx().x-1)*CUDA.blockDim().x + CUDA.threadIdx().x

@device_override @inline function __get_indices(::Val{2})
    ix = (CUDA.blockIdx().x-1)*CUDA.blockDim().x + CUDA.threadIdx().x
    iy = (CUDA.blockIdx().y-1)*CUDA.blockDim().y + CUDA.threadIdx().y
    return ix, iy
end

@device_override @inline function __get_indices(::Val{3})
    ix = (CUDA.blockIdx().x-1)*CUDA.blockDim().x + CUDA.threadIdx().x
    iy = (CUDA.blockIdx().y-1)*CUDA.blockDim().y + CUDA.threadIdx().y
    iz = (CUDA.blockIdx().z-1)*CUDA.blockDim().z + CUDA.threadIdx().z
    return ix, iy, iz
end

end
