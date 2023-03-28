using BenchmarkTools

function compute!(A, B, C, s)
    for ix ∈ CartesianIndices(A)
        @inbounds A[ix] = B[ix] + s * C[ix]
    end
    return
end

function main(; nx=128, ny=128)
    B = rand(Float64, nx, ny)
    C = ones(Float64, nx, ny)
    A = similar(B)
    s = 2.0
    compute!(A, B, C, s)
    t_it = @belapsed compute!($A, $B, $C, $s)
    T_eff = 3 * prod(size(B)) * sizeof(eltype(B)) * 1e-9 / t_it
    println("Perf: $(round(T_eff)) GB/s ($t_it sec)")
    return #t_it, T_eff
end

n = 1024
main(; nx=n, ny=n)