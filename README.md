<h1> <img src="docs/logo/logo_TinyKernels.png" alt="TinyKernels.jl" width="50"> TinyKernels.jl </h1>

[![CI](https://github.com/utkinis/TinyKernels.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/utkinis/TinyKernels.jl/actions/workflows/CI.yml)

**TinyKernels.jl** provides a tiny abstraction for GPU (and CPU) kernels, with full support for CUDA (Nvidia) and ROCm (AMD) backends, limited support for Metal (GPU programming on MacOS ARM) backend, and allowing for multi-threaded CPU execution.

TinyKernels.jl is mostly a heavily stripped-down version of [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) supporting the bare minimum of the features. This package provides a sandbox for Julia GPU tooling and to measure the performance of kernels in a GPU-agnostic way. While the API of KernelAbstractions.jl is in a "transient" state, this package will provide the thin abstraction layer on top the [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl), [AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl) and [Metal.jl](https://github.com/JuliaGPU/Metal.jl) packages.

TinyKernels.jl allows to explicitly launch GPU kernels asynchronously on different streams with given priority. This feature facilitates the overlap between computations and memory transfers in distributed configurations.

TinyKernels.jl supports automatic differentiation with [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl) overloading the `Enzyme.autodiff` function to enable reverse mode AD of GPU (and CPU) kernels.

Preliminary benchmarks can be found in [TinyBenchmarks.jl](https://github.com/luraess/TinyBenchmarks.jl) and Metal playground in [MetalGPU](https://github.com/luraess/MetalGPU).

Stay tuned :rocket:

### Notes

⚠️ **Metal backend:**
- Only `Float32` is being supported. For `Float64`, one could try using a construct from [DoubleFloats.jl](https://github.com/JuliaMath/DoubleFloats.jl/blob/ef689ccbab37d84943e2533309d34c6665229cab/src/Double.jl#L30) _which may impact performance_.