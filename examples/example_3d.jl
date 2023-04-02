# example triad 2D kernel
using TinyKernels
using TinyKernels.CPUBackend

# Select based upon your local device (:CUDA, :AMDGPU, :Metal)
run = :none

@static if run == :CUDA
    using CUDA
    CUDA.functional() && (using TinyKernels.CUDABackend)
    device = CUDADevice()
elseif run == :AMDGPU
    using AMDGPU
    AMDGPU.functional() && (using TinyKernels.ROCBackend)
    device = ROCBackend.ROCDevice()
elseif run == :Metal
    using Metal
    Metal.functional() && (using TinyKernels.MetalBackend)
    device = MetalDevice()
end

@tiny function triad_3d!(A, B, C, s)
    ix, iy, iz = @indices()
    for _ in 1:10
        @inbounds A[ix, iy, iz] = B[ix, iy, iz] + s * C[ix, iy, iz]
    end
    return
end

function main(::Type{DAT}; device) where DAT
    nx, ny, nz = 256, 256, 256
    A = device_array(DAT, device, nx, ny, nz)
    B = device_array(DAT, device, nx, ny, nz)
    C = device_array(DAT, device, nx, ny, nz)

    fill!(B, DAT(1.0))
    fill!(C, DAT(2.0))

    s = DAT(-1.0)

    ranges = ((4:nx-3 , 4:ny-3 , 4:nz-3 ),
              (1:3    , 1:ny   , 4:nz-3 ),
              (nx-2:nx, 1:ny   , 4:nz-3 ),
              (4:nx-3 , 1:3    , 4:nz-3 ),
              (4:nx-3 , ny-2:ny, 4:nz-3 ),
              (1:nx   , 1:ny   , 1:3    ),
              (1:nx   , 1:ny   , nz-2:nz))

    kernel_triad_3d! = triad_3d!(device)

    TinyKernels.device_synchronize(device)
    for it in 1:100
        println("  step $it")
        inner_event  =  kernel_triad_3d!(A, B, C, s; ndrange=ranges[1])
        outer_events = [kernel_triad_3d!(A, B, C, s; ndrange=ranges[i], priority=:high) for i in 2:lastindex(ranges)]

        wait(outer_events)
        # sleep(1/30)
        wait(inner_event)
    end

    @assert A ≈ B .+ s .* C

    return
end

@static if run == :CUDA
    println("running on CUDA device...")
    main(Float64; device)
elseif run == :AMDGPU
    println("running on AMD device...")
    main(Float64; device)
elseif run == :Metal
    println("running on Metal device...")
    main(Float32; device)
end

println("running on CPU device...")
main(Float64; device=CPUDevice())