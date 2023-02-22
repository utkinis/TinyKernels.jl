<h1> <img src="docs/logo/logo_TinyKernels.png" alt="TinyKernels.jl" width="50"> TinyKernels.jl </h1>

[![CI](https://github.com/utkinis/TinyKernels.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/utkinis/TinyKernels.jl/actions/workflows/CI.yml)

`TinyKernels.jl` provides a tiny absraction for GPU kernels, currently supporting CUDA (Nvidia) and ROCm (AMD) backends.

Currently, `TinyKernels.jl` is mostly a heavily stripped down version of [`KernelAbstractions.jl`](https://github.com/JuliaGPU/KernelAbstractions.jl) supporting the bare minimum of the features. I made this package to learn about Julia and to measure the performance of kernels in a GPU-agnostic way. While the API of `KernelAbstractions.jl` is in a transient state, this package will provide the thin abstraction layer on top the [`CUDA.jl`](https://github.com/JuliaGPU/CUDA.jl) and [`AMDGPU.jl`](https://github.com/JuliaGPU/AMDGPU.jl) packages.