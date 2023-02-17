using TinyKernels

using TinyKernels.ROCBackend

using AMDGPU

function test_function!(range, A, B, C, s)
    ix = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x + (range[1][1] - 1)
    iy = (workgroupIdx().y - 1) * workgroupDim().y + workitemIdx().y + (range[2][1] - 1)
    if ix ∈ axes(A, 1) && iy ∈ axes(A, 2)
        for _ in 1:10
            @inbounds A[ix, iy] = B[ix, iy] + s * C[ix, iy]
        end
    end
    return
end

function main()
    nx, ny = 4096, 4096
    A = AMDGPU.zeros(Float64, nx, ny)
    B = AMDGPU.ones(Float64, nx, ny)
    C = 2.0 .* AMDGPU.ones(Float64, nx, ny)
    s = -1.0

    ranges = ((4:nx-3 , 4:ny-3 ),
              (1:3    , 1:ny   ),
              (nx-2:nx, 1:ny   ),
              (4:nx-3 , 1:3    ),
              (4:nx-3 , ny-2:ny))

    test_kernel! = Kernel(test_function!, ROCBackend.ROCDevice(), ranges)

    # Doens't seem to be necessary
    # event = AMDGPU.barrier_and!(AMDGPU.default_queue(), AMDGPU.active_kernels(AMDGPU.default_queue()))
    # wait(event)

    inner_event, outer_events... = test_kernel!(A, B, C, s)

    wait(outer_events)
    sleep(1 / 30)
    wait(inner_event)

    @assert A ≈ B .+ s .* C

    return
end

for i in 1:5
    main()
end
