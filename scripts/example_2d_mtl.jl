using TinyKernels

using Metal

@static if !isnothing(Metal.devices())
    using TinyKernels.MetalBackend
end

@tiny function kernel_test_2d!(A, B, C, s)
    ix, iy = @indices()
    for _ in 1:10
        @inbounds A[ix, iy] = B[ix, iy] + s * C[ix, iy]
    end
    return
end

function main(; device)
    nx, ny = 128, 256
    A = device_array(Float32, device, nx, ny)
    B = device_array(Float32, device, nx, ny)
    C = device_array(Float32, device, nx, ny)

    fill!(B,1.0)
    fill!(C,2.0)

    s = -1.0f0

    ranges = ((4:nx-3 , 4:ny-3 ),
              (1:3    , 1:ny   ),
              (nx-2:nx, 1:ny   ),
              (4:nx-3 , 1:3    ),
              (4:nx-3 , ny-2:ny))

    test! = Kernel(kernel_test_2d!, device)

    TinyKernels.device_synchronize(device)
    for it in 1:100
        println("  step $it")
        # ev = test!(A,B,C,s; ndrange=size(A))
        # wait(ev)
        inner_event  =  test!(A,B,C,s; ndrange=ranges[1])
        outer_events = [test!(A,B,C,s; ndrange=ranges[i]) for i in 2:lastindex(ranges)]
        wait(outer_events)
        # sleep(1/30)
        wait(inner_event)
    end

    @assert A ≈ B .+ s .* C
    return
end

@static if !isnothing(Metal.devices())
    println("running on Metal device...")
    main(;device=MetalDevice())
end
