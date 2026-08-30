using LinearAlgebra
using Printf
using FFTW
using Plots

# ------------------------------------------------------------
# Problem definition
#
#   -Δu = f    in Ω = (0,1) × (0,1)
#      u = 0   on ∂Ω
#
# Exact solution:
#
#   u(x,y) = sin(πx) sin(πy)
#
# Therefore
#
#   f(x,y) = 2π² sin(πx) sin(πy)
#
# Homogeneous Dirichlet on a rectangle → DST-I, implemented as an
# unnormalized FFT of the odd extension (same algorithm as
# rust_tenferro_fft). This is the exact inverse of the 5-point Laplacian.
# ------------------------------------------------------------

u_exact(x, y) = sin(pi * x) * sin(pi * y)
f(x, y) = 2pi^2 * sin(pi * x) * sin(pi * y)

"""
Odd extension of length `2(M+1)` along dimension 1.

`s[1]=0`, `s[2:M+1]=x`, `s[L+1]=0`, `s[L+2:2L]=-reverse(x)`, `L=M+1`.
"""
function odd_extend_dim1(x)
    M, ncols = size(x)
    L = M + 1
    s = zeros(eltype(x), 2L, ncols)
    s[2:M+1, :] .= x
    s[L+2:2L, :] .= .-reverse(x, dims=1)
    return s
end

function odd_extend_dim2(x)
    nrows, M = size(x)
    L = M + 1
    s = zeros(eltype(x), nrows, 2L)
    s[:, 2:M+1] .= x
    s[:, L+2:2L] .= .-reverse(x, dims=2)
    return s
end

"""
1-D DST-I along dimension 1 via FFT of the odd extension.

Forward: `S_k = -Im(FFT(s)[k])` for `k = 1:M`.
"""
function dst_dim1(x)
    M = size(x, 1)
    Y = fft(odd_extend_dim1(x), 1)
    return -imag.(Y[2:M+1, :])
end

function dst_dim2(x)
    M = size(x, 2)
    Y = fft(odd_extend_dim2(x), 2)
    return -imag.(Y[:, 2:M+1])
end

function dst2d(x)
    return dst_dim2(dst_dim1(x))
end

"""Exact solve of the 5-point Dirichlet Poisson system by 2-D DST-I."""
function solve_dst(rhs, h)
    N = size(rhs, 1)
    M = N - 2
    L = M + 1
    @assert L == N - 1

    interior = rhs[2:N-1, 2:N-1]
    fhat = dst2d(interior)

    # λ_{p,q} = [2-2cos(pπ/L) + 2-2cos(qπ/L)] / h², p,q = 1:M.
    p = 1:M
    λx = 2 .- 2 .* cos.(p .* pi ./ L)
    λy = 2 .- 2 .* cos.(p .* pi ./ L)
    λ = (λx .+ λy') ./ h^2
    uhat = fhat ./ λ

    uint = dst2d(uhat) ./ (2L)^2

    u = zeros(eltype(rhs), N, N)
    u[2:N-1, 2:N-1] .= uint
    return u
end

function save_plot(path, x, y, u, ue, err)
    p1 = surface(
        x,
        y,
        ue',
        xlabel="x",
        ylabel="y",
        zlabel="u",
        title="Exact solution",
        camera=(45, 30)
    )

    p2 = surface(
        x,
        y,
        u',
        xlabel="x",
        ylabel="y",
        zlabel="u",
        title="Numerical solution",
        camera=(45, 30)
    )

    p3 = heatmap(
        x,
        y,
        err',
        xlabel="x",
        ylabel="y",
        title="Absolute error",
        colorbar_title="|u - u_exact|"
    )

    p = plot(
        p1,
        p2,
        p3,
        layout=(1, 3),
        size=(1500, 450)
    )

    display(p)
    savefig(p, path)
    return p
end

function main()
    N = 401

    # Match the single-thread Rust comparison.
    FFTW.set_num_threads(1)
    BLAS.set_num_threads(1)

    x = range(0.0, 1.0, length=N)
    y = range(0.0, 1.0, length=N)
    h = 1.0 / (N - 1)

    @printf("N = %d\n", N)
    @printf("h = %.6e\n", h)

    rhs = zeros(Float64, N, N)
    for j in 1:N
        for i in 1:N
            rhs[i, j] = f(x[i], y[j])
        end
    end

    start_time = time()
    u = solve_dst(rhs, h)
    end_time = time()
    duration = end_time - start_time

    println()
    @printf("time = %.6f seconds\n", duration)

    ue = zeros(Float64, N, N)
    for j in 1:N
        for i in 1:N
            ue[i, j] = u_exact(x[i], y[j])
        end
    end

    err = abs.(u .- ue)
    max_error = maximum(err)
    l2_error = sqrt(sum((u .- ue).^2) * h^2)

    println()
    @printf("max error = %.6e\n", max_error)
    @printf("L2 error  = %.6e\n", l2_error / sqrt(N))

    out = joinpath(@__DIR__, "poisson_fft.png")
    save_plot(out, x, y, u, ue, err)
    println("saved ", out)

    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
