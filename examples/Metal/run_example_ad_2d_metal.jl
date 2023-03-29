using Metal
using Enzyme
using .TinyKernels
using .TinyKernels.MetalBackend
using .TinyKernels.KernelAD

include("../example_ad_2d.jl")

@static if Metal.functional()
    println("running on Metal device...")
    main(Float32; device=MetalDevice())
end