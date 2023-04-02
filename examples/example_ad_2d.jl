# example triad 2D kernel
using Enzyme
using TinyKernels
using TinyKernels.CPUBackend
using TinyKernels.KernelAD

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

@tiny function triad_2d!(A, B, C, s)
    ix, iy = @indices()
    for _ in 1:10
        @inbounds A[ix, iy] = B[ix, iy] + s * C[ix, iy]
    end
    return
end

function main(::Type{DAT}; device) where DAT
    nx, ny = 32, 32
    A = device_array(DAT, device, nx, ny)
    B = device_array(DAT, device, nx, ny)
    C = device_array(DAT, device, nx, ny)

    fill!(B, DAT(1.0))
    fill!(C, DAT(2.0))

    s = DAT(-1.0)

    kernel_triad_2d! = triad_2d!(device)

    grad_kernel_triad_2d! = Enzyme.autodiff(kernel_triad_2d!)
    dA = copy(A); fill!(dA, DAT(1.0))
    dB = copy(A)

    TinyKernels.device_synchronize(device)

    wait(grad_kernel_triad_2d!(DuplicatedNoNeed(A, dA), DuplicatedNoNeed(B, dB), Const(C), Const(s); ndrange=size(A)))
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