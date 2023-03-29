using AMDGPU
using Enzyme
using .TinyKernels
using .TinyKernels.ROCBackend
using .TinyKernels.KernelAD

include("../example_ad_2d.jl")

@static if AMDGPU.functional()
    println("running on AND device...")
    main(Float64; device=ROCBackend.ROCDevice())
end