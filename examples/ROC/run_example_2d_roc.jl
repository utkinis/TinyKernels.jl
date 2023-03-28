using AMDGPU
using .TinyKernels
using .TinyKernels.MetalBackend
using .TinyKernels.CPUBackend

include("../example_2d.jl")

@static if AMDGPU.functional()
    println("running on AND device...")
    main(Float64; device=ROCBackend.ROCDevice())
end

println("running on CPU device...")
main(Float64; device=CPUDevice())