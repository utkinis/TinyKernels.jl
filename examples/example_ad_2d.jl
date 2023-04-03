# example triad 2D kernel
using Enzyme
using TinyKernels
using TinyKernels.KernelAD

include("setup_example.jl")

# Select based upon your local device (:CPU, :CUDA, :AMDGPU, :Metal)
backend = :CPU

@setup_example()

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

println("running on $backend device...")
main(eletype; device)