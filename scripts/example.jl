using TinyKernels

using CUDA

@static if CUDA.functional()
    using TinyKernels.CUDABackend
end

using AMDGPU

@static if AMDGPU.functional()
    using TinyKernels.ROCBackend
end

@tiny function kernel_test!(A, B, C, s)
    ix, iy = @indices()
    for _ in 1:10
        @inbounds A[ix, iy] = B[ix, iy] + s * C[ix, iy]
    end
    return
end

function main(; device)
    nx, ny = 4096, 4096
    A = device_array(Float64, device, nx, ny)
    B = device_array(Float64, device, nx, ny)
    C = device_array(Float64, device, nx, ny)

    fill!(B,1.0)
    fill!(C,2.0)

    s = -1.0

    ranges = ((4:nx-3 , 4:ny-3 ),
              (1:3    , 1:ny   ),
              (nx-2:nx, 1:ny   ),
              (4:nx-3 , 1:3    ),
              (4:nx-3 , ny-2:ny))

    test! = Kernel(kernel_test!, device)

    synchronize()
    for i in 1:100
        println("step $i")
        inner_event = test!(view(A, ranges[1]...),
                            view(B, ranges[1]...),
                            view(C, ranges[1]...), s; range=length.(ranges[1]))
        outer_events = [test!(view(A, ranges[i]...),
                              view(B, ranges[i]...),
                              view(C, ranges[i]...), s; range=length.(ranges[i]), priority=:high) for i in 2:lastindex(ranges)]

        wait(outer_events)
        # sleep(1/30)
        wait(inner_event)
    end

    @assert A ≈ B .+ s .* C

    return
end

main(;device=CUDADevice())
