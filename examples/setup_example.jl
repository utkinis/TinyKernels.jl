macro setup_example()
    esc(quote
        @assert backend ∈ (:CPU, :CUDA, :AMDGPU, :Metal) "backend must be one of (:CPU, :CUDA, :AMDGPU, :Metal), got :$backend"
        # import backend-specific API
        @static if backend == :CPU
            using TinyKernels.CPUBackend
            device = CPUDevice()
        elseif backend == :CUDA
            using CUDA
            @assert CUDA.functional(); using TinyKernels.CUDABackend
            device = CUDADevice()
        elseif backend == :AMDGPU
            using AMDGPU
            @assert AMDGPU.functional(); using TinyKernels.ROCBackend
            device = ROCBackend.ROCDevice()
        elseif backend == :Metal
            using Metal
            @assert Metal.functional(); using TinyKernels.MetalBackend
            device = MetalDevice()
        end
        # Metal doesn't support Float64
        @static if backend == :Metal
            eletype = Float32
        else
            eletype = Float64
        end
    end)
end