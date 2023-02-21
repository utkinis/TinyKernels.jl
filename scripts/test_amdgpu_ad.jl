using TinyKernels
using TinyKernels.ROCBackend
using TinyKernels.KernelAD

using AMDGPU
using Enzyme
using Random

function test_function!(range, RUx, RUy, Ux, Uy)
    __ix = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    __iy = (workgroupIdx().y - 1) * workgroupDim().y + workitemIdx().y
    if __ix > length(range[1]) || __iy > length(range[2]) return end
    ix, iy = range[1][__ix], range[2][__iy]
    if ix ∈ axes(RUx, 1)[2:end-1] && iy ∈ axes(RUx, 2)[2:end-1]
        @inbounds RUx[ix, iy] = Ux[ix-1, iy]^2 - 2.0 * Ux[ix, iy]^2 + Ux[ix+1, iy]^2 + 0.5 * Uy[ix, iy]^2
    end
    if ix ∈ axes(RUy, 1)[2:end-1] && iy ∈ axes(RUy, 2)[2:end-1]
        @inbounds RUy[ix, iy] = Uy[ix, iy-1]^2 - 2.0 * Uy[ix, iy]^2 + Uy[ix, iy+1]^2 + 0.5 * Ux[ix, iy]^2
    end
    return
end

function main()
    # AMDGPU.default_device_id!(7)
    Random.seed!(24041994)
    nx, ny = 4, 5
    Ux  = AMDGPU.rand(Float64, nx, ny)
    Uy  = AMDGPU.rand(Float64, nx, ny)
    RUx = AMDGPU.zeros(Float64, nx, ny)
    RUy = AMDGPU.zeros(Float64, nx, ny)
    # AD VJP
    ∂Rx_∂R = AMDGPU.ones(Float64, nx, ny)
    ∂Ry_∂R = AMDGPU.ones(Float64, nx, ny)
    ∂Ux_∂R = AMDGPU.zeros(Float64, nx, ny)
    ∂Uy_∂R = AMDGPU.zeros(Float64, nx, ny)
    # Exact VJP
    ∂Ux_∂R_exact = AMDGPU.zeros(Float64, nx, ny)
    ∂Ux_∂R_exact[[1, end], 2:end-1] .= 2.0 .* Ux[[1, end], 2:end-1]
    ∂Ux_∂R_exact[2:end-1, 2:end-1] .= Ux[2:end-1, 2:end-1]
    # Generate kernel
    test_kernel! = Kernel(test_function!, ROCBackend.ROCDevice())
    # Generate kernel gradient
    grad_test_kernel! = Enzyme.autodiff(test_kernel!)
    # Evaluate forward problem
    wait(test_kernel!(RUx, RUy, Ux, Uy; range=size(Ux)))
    @show RUx
    # Compute VJP
    wait(grad_test_kernel!(Duplicated(RUx, ∂Rx_∂R), Duplicated(RUy, ∂Ry_∂R), Duplicated(Ux, ∂Ux_∂R), Duplicated(Uy, ∂Uy_∂R); range=size(Ux)))
    @show ∂Ux_∂R ∂Ux_∂R_exact
    @show ∂Rx_∂R
    return
end

main()
