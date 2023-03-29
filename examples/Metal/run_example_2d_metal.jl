using Metal
using .TinyKernels
using .TinyKernels.MetalBackend

include("../example_2d.jl")

@static if Metal.functional()
    println("running on Metal device...")
    main(Float32; device=MetalDevice())
end