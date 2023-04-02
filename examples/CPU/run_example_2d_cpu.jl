using .TinyKernels
using .TinyKernels.CPUBackend

include("../example_2d.jl")

println("running on CPU device...")
main(Float64; device=CPUDevice())