using Test
using TinyKernels

device = CPUDevice()
DAT = Float64
nx, ny, nz = 10, 11, 12

@tiny function triad_1d!(A, B, C, s)
    ix, = @indices()
    @inbounds A[ix] = B[ix] + s * C[ix]
    return
end

@tiny function triad_2d!(A, B, C, s)
    ix, iy = @indices()
    @inbounds A[ix, iy] = B[ix, iy] + s * C[ix, iy]
    return
end

@tiny function triad_3d!(A, B, C, s)
    ix, iy, iz = @indices()
    @inbounds A[ix, iy, iz] = B[ix, iy, iz] + s * C[ix, iy, iz]
    return
end

@tiny function kernel_test!(RUx, RUy, Ux, Uy)
    ix, iy = @indices()
    if ix ∈ axes(RUx, 1) && iy ∈ axes(RUx, 2)
        @inbounds RUx[ix, iy] = Ux[ix, iy]^2 - convert(eltype(Ux), 2.0) * Ux[ix+1, iy]^2 + Ux[ix+2, iy]^2 + convert(eltype(Ux), 0.5) * Uy[ix+1, iy]^2
    end
    if ix ∈ axes(RUy, 1) && iy ∈ axes(RUy, 2)
        @inbounds RUy[ix, iy] = Uy[ix, iy]^2 - convert(eltype(Uy), 2.0) * Uy[ix, iy+1]^2 + Uy[ix, iy+2]^2 + convert(eltype(Uy), 0.5) * Ux[ix, iy+1]^2
    end
    return
end

kernel_triad_1d! = triad_1d!(device)
kernel_triad_2d! = triad_2d!(device)
kernel_triad_3d! = triad_3d!(device)

A1 = device_array(DAT, device, nx)
B1 = device_array(DAT, device, nx)
C1 = device_array(DAT, device, nx)

A2 = device_array(DAT, device, nx, ny)
B2 = device_array(DAT, device, nx, ny)
C2 = device_array(DAT, device, nx, ny)

A3 = device_array(DAT, device, nx, ny, nz)
B3 = device_array(DAT, device, nx, ny, nz)
C3 = device_array(DAT, device, nx, ny, nz)

fill!(B1, DAT(1.0))
fill!(B2, DAT(1.0))
fill!(B3, DAT(1.0))
fill!(C1, DAT(2.0))
fill!(C2, DAT(2.0))
fill!(C3, DAT(2.0))
s = DAT(-1.0)

ranges1 = ((4:nx-3 ),
           (1:3    ),
           (nx-2:nx))

ranges2 = ((4:nx-3 , 4:ny-3 ),
           (1:3    , 1:ny   ),
           (nx-2:nx, 1:ny   ),
           (4:nx-3 , 1:3    ),
           (4:nx-3 , ny-2:ny))

ranges3 = ((4:nx-3 , 4:ny-3 , 4:nz-3 ),
           (1:3    , 1:ny   , 4:nz-3 ),
           (nx-2:nx, 1:ny   , 4:nz-3 ),
           (4:nx-3 , 1:3    , 4:nz-3 ),
           (4:nx-3 , ny-2:ny, 4:nz-3 ),
           (1:nx   , 1:ny   , 1:3    ),
           (1:nx   , 1:ny   , nz-2:nz))

@testset "triad (saxpy)" begin
    TinyKernels.device_synchronize(device)
    @testset "triad 1D" begin
        inn_ev =  kernel_triad_1d!(A1, B1, C1, s; ndrange=ranges1[1])
        out_ev = [kernel_triad_1d!(A1, B1, C1, s; ndrange=ranges1[i], priority=:high) for i in 2:lastindex(ranges1)]
        wait(out_ev)
        sleep(1/30) # could be MPI comm
        wait(inn_ev)
        @test A1 ≈ B1 .+ s .* C1
    end

    TinyKernels.device_synchronize(device)
    @testset "triad 1D" begin
        inn_ev =  kernel_triad_2d!(A2, B2, C2, s; ndrange=ranges2[1])
        out_ev = [kernel_triad_2d!(A2, B2, C2, s; ndrange=ranges2[i], priority=:high) for i in 2:lastindex(ranges2)]
        wait(out_ev)
        sleep(1/30) # could be MPI comm
        wait(inn_ev)
        @test A2 ≈ B2 .+ s .* C2
    end

    TinyKernels.device_synchronize(device)
    @testset "triad 1D" begin
        inn_ev =  kernel_triad_3d!(A3, B3, C3, s; ndrange=ranges3[1])
        out_ev = [kernel_triad_3d!(A3, B3, C3, s; ndrange=ranges3[i], priority=:high) for i in 2:lastindex(ranges3)]
        wait(out_ev)
        sleep(1/30) # could be MPI comm
        wait(inn_ev)
        @test A3 ≈ B3 .+ s .* C3
    end
end