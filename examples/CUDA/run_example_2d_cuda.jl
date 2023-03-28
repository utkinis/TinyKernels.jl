using CUDA
using .TinyKernels
using .TinyKernels.MetalBackend
using .TinyKernels.CPUBackend

include("../example_2d.jl")

@static if CUDA.functional()
    println("running on CUDA device...")
    main(Float64; device=CUDADevice())
end

println("running on CPU device...")
main(Float64; device=CPUDevice())