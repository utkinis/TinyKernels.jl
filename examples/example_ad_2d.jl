using TinyKernels
using TinyKernels.KernelAD
using Enzyme

using CUDA

@static if CUDA.functional()
    using TinyKernels.CUDABackend
end

using AMDGPU

@static if AMDGPU.functional()
    using TinyKernels.ROCBackend
end

using TinyKernels.CPUBackend

@tiny function test_function!(A, B, C, s)
    ix, iy = @indices()
    for _ in 1:10
        @inbounds A[ix, iy] = B[ix, iy] + s * C[ix, iy]
    end
    return
end

function main(; device)
    nx, ny = 32, 32
    A = device_array(Float64, device, nx, ny)
    B = device_array(Float64, device, nx, ny)
    C = device_array(Float64, device, nx, ny)

    fill!(B,1.0)
    fill!(C,2.0)

    s = -1.0

    test_kernel! = test_function!(device)

    grad_test_kernel! = Enzyme.autodiff(test_kernel!)
    dA = copy(A); fill!(dA,1.0)
    dB = copy(A)

    TinyKernels.device_synchronize(device)

    wait(grad_test_kernel!(DuplicatedNoNeed(A, dA), DuplicatedNoNeed(B, dB), Const(C), Const(s); ndrange=size(A)))
    return
end

@static if CUDA.functional()
    println("running on CUDA device...")
    main(; device=CUDADevice())
end

@static if AMDGPU.functional()
    println("running on AMD device...")
    main(; device=ROCBackend.ROCDevice())
end

println("running on CPU device...")
main(; device=CPUDevice())