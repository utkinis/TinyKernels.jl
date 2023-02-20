using TinyKernels
using TinyKernels.CUDABackend
using TinyKernels.KernelAD

using CUDA
using Enzyme

function test_function!(range, A, B, C, s)
    __ix = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    __iy = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    if __ix > length(range[1]) || __iy > length(range[2]) return end
    ix,iy = range[1][__ix], range[2][__iy]
    if ix ∈ axes(A, 1) && iy ∈ axes(A, 2)
        for _ in 1:10
            @inbounds A[ix, iy] = B[ix, iy] + s * C[ix, iy]
        end
    end
    return
end

function main()
    nx, ny = 32, 32
    A = CUDA.zeros(Float64, nx, ny)
    B = CUDA.ones(Float64, nx, ny)
    C = 2.0.*CUDA.ones(Float64, nx, ny)
    s = -1.0

    ranges = ((4:nx-3 , 4:ny-3 ),
              (1:3    , 1:ny   ),
              (nx-2:nx, 1:ny   ),
              (4:nx-3 , 1:3    ),
              (4:nx-3 , ny-2:ny))

    test_kernel! = Kernel(test_function!, CUDADevice())

    grad_test_kernel! = Enzyme.autodiff(test_kernel!)
    dA = copy(A)
    dB = copy(A)

    synchronize()
    wait(grad_test_kernel!(Duplicated(A, dA), DuplicatedNoNeed(B, dB), Const(C), Const(s); range=size(A)))

    synchronize()
    for i in 1:100
        println("step $i")
        inner_event  =  test_kernel!(A, B, C, s; range=ranges[1])
        outer_events = [test_kernel!(A, B, C, s; range=ranges[i], priority=:high) for i in 2:lastindex(ranges)]

        wait(outer_events)
        sleep(1/30)
        wait(inner_event)
    end

    @assert A ≈ B .+ s .* C

    return
end

main()
