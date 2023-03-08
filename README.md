<h1> <img src="docs/logo/logo_TinyKernels.png" alt="TinyKernels.jl" width="50"> TinyKernels.jl </h1>

[![CI](https://github.com/utkinis/TinyKernels.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/utkinis/TinyKernels.jl/actions/workflows/CI.yml)

**TinyKernels.jl** provides a tiny abstraction for GPU kernels, with full support for CUDA (Nvidia) and ROCm (AMD) backends, and limited support for Metal (GPU programming on MacOS ARM).

TinyKernels.jl is mostly a heavily stripped-down version of [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) supporting the bare minimum of the features. This package provides a sandbox for Julia GPU tooling and to measure the performance of kernels in a GPU-agnostic way. While the API of KernelAbstractions.jl is in a "transient" state, this package will provide the thin abstraction layer on top the [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl), [AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl) and [Metal.jl](https://github.com/JuliaGPU/Metal.jl) packages.

TinyKernels.jl allows to explicitly launch GPU kernels asynchronously on different streams or queues with given priority. This feature facilitates the overlap between computations and memory transfers in distributed configurations.

Preliminary benchmarks can be found [TinyBenchmarks.jl](https://github.com/luraess/TinyBenchmarks.jl).

Stay tuned :rocket:
