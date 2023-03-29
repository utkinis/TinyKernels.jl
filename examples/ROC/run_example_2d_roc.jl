using AMDGPU
using .TinyKernels
using .TinyKernels.ROCBackend

include("../example_2d.jl")

@static if AMDGPU.functional()
    println("running on AND device...")
    main(Float64; device=ROCBackend.ROCDevice())
end