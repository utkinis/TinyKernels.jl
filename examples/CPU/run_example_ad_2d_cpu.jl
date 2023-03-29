using Enzyme
using .TinyKernels
using .TinyKernels.CPUBackend
using .TinyKernels.KernelAD

include("../example_ad_2d.jl")

println("running on CPU device...")
main(Float64; device=CPUDevice())