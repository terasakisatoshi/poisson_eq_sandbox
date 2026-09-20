using Printf
using Plots
using LoopVectorization

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
# The hot sweeps are expanded with `LoopVectorization.@turbo`, which emits
# SIMD code for the inner loop; the tracking sweeps use a plain `@inbounds`
# loop so the update-width reduction is computed. Buffer are swapped
# instead of copied; both stay zero on the Dirichlet boundary because only
# interior points are written. The update-width reduction and the progress
# line run every 1000th sweep, not every sweep.
# ------------------------------------------------------------

u_exact(x, y) = sin(pi * x) * sin(pi * y)
f(x, y) = 2pi^2 * sin(pi * x) * sin(pi * y)

"""
One Jacobi sweep over the interior using the five-point stencil.

When `TRACK` is `false` the inner loop is macro-expanded with `@turbo`.
When `TRACK` is `true` a plain `@inbounds` loop computes the update-width
reduction; only every 1000th sweep tracks the error (see `jacobi!`).
"""
function jacobi_sweep!(u, u_new, rhs, h2, N, ::Val{TRACK}) where {TRACK}
    update_error = 0.0

    # Interior points only.
    # Boundary points remain zero (Dirichlet).
    if TRACK
        @inbounds for j in 2:N-1
            for i in 2:N-1
                val = 0.25 * (
                    u[i+1, j] +
                    u[i-1, j] +
                    u[i, j+1] +
                    u[i, j-1] +
                    h2 * rhs[i, j]
                )
                update_error = max(update_error, abs(val - u[i, j]))
                u_new[i, j] = val
            end
        end
    else
        @inbounds for j in 2:N-1
            @turbo for i in 2:N-1
                val = 0.25 * (
                    u[i+1, j] +
                    u[i-1, j] +
                    u[i, j+1] +
                    u[i, j-1] +
                    h2 * rhs[i, j]
                )
                u_new[i, j] = val
            end
        end
    end

    return update_error
end

function jacobi!(u, u_new, rhs, h, tol, maxiter)
    N = size(u, 1)
    update_error = Inf
    iterations = 0
    h2 = h^2

    for iter in 1:maxiter
        # Track the update width (and print) only at reported iterations,
        # so the hot sweeps stay free of the reduction and the output call.
        report = iter % 1000 == 0 || iter == maxiter
        if report
            update_error = jacobi_sweep!(u, u_new, rhs, h2, N, Val(true))
        else
            jacobi_sweep!(u, u_new, rhs, h2, N, Val(false))
        end

        u, u_new = u_new, u
        iterations = iter

        if report
            @printf(
                "iteration = %6d, update error = %.6e\n",
                iter,
                update_error
            )
            if update_error < tol
                break
            end
        end
    end

    return u, iterations, update_error
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
    tol = 1e-10
    maxiter = 100_000

    x = range(0.0, 1.0, length=N)
    y = range(0.0, 1.0, length=N)
    h = 1.0 / (N - 1)

    @printf("N = %d\n", N)
    @printf("h = %.6e\n", h)

    u = zeros(Float64, N, N)
    u_new = zeros(Float64, N, N)
    rhs = zeros(Float64, N, N)
    @inbounds for j in 1:N
        for i in 1:N
            rhs[i, j] = f(x[i], y[j])
        end
    end

    start_time = time()
    u, iterations, update_error = jacobi!(u, u_new, rhs, h, tol, maxiter)
    end_time = time()
    duration = end_time - start_time

    println()
    @printf("time = %.6f seconds\n", duration)
    @printf("Jacobi iterations = %d\n", iterations)
    @printf("final update error = %.6e\n", update_error)

    ue = zeros(Float64, N, N)
    @inbounds for j in 1:N
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

    out = joinpath(@__DIR__, "poisson_jacobi.png")
    save_plot(out, x, y, u, ue, err)
    println("saved ", out)

    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
