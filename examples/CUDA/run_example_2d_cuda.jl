using CUDA
using .TinyKernels
using .TinyKernels.CUDABackend

include("../example_2d.jl")

@static if CUDA.functional()
    println("running on CUDA device...")
    main(Float64; device=CUDADevice())
end