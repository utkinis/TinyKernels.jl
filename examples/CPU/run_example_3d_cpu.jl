using .TinyKernels
using .TinyKernels.CPUBackend

include("../example_3d.jl")

println("running on CPU device...")
main(Float64; device=CPUDevice())