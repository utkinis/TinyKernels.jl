using TinyKernels

using CUDA

@static if CUDA.functional()
    using TinyKernels.CUDABackend
end

using AMDGPU

@static if AMDGPU.functional()
    using TinyKernels.ROCBackend
end

@tiny function kernel_test_3d!(A, B, C, s)
    ix, iy, iz = @indices()
    for _ in 1:10
        @inbounds A[ix, iy, iz] = B[ix, iy, iz] + s * C[ix, iy, iz]
    end
    return
end

function main(; device)
    nx, ny, nz = 256, 256, 256
    A = device_array(Float64, device, nx, ny, nz)
    B = device_array(Float64, device, nx, ny, nz)
    C = device_array(Float64, device, nx, ny, nz)

    fill!(B,1.0)
    fill!(C,2.0)

    s = -1.0

    ranges = ((4:nx-3 , 4:ny-3 , 4:nz-3 ),
              (1:3    , 1:ny   , 4:nz-3 ),
              (nx-2:nx, 1:ny   , 4:nz-3 ),
              (4:nx-3 , 1:3    , 4:nz-3 ),
              (4:nx-3 , ny-2:ny, 4:nz-3 ),
              (1:nx   , 1:ny   , 1:3    ),
              (1:nx   , 1:ny   , nz-2:nz))

    test! = Kernel(kernel_test_3d!, device)

    sleep(1) # workaround to aviod synchronize device
    for i in 1:100
        println("  step $i")
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

@static if CUDA.functional()
    println("running on CUDA device...")
    main(;device=CUDADevice())
end

@static if AMDGPU.functional()
    println("running on AMD device...")
    main(;device=ROCBackend.ROCDevice())
end