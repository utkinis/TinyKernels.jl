using CUDA
using Enzyme
using .TinyKernels
using .TinyKernels.CUDABackend
using .TinyKernels.KernelAD

include("../example_ad_2d.jl")

@static if CUDA.functional()
    println("running on CUDA device...")
    main(Float64; device=CUDADevice())
end