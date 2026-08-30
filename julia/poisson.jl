using Printf
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
# ------------------------------------------------------------

u_exact(x, y) = sin(pi * x) * sin(pi * y)
f(x, y) = 2pi^2 * sin(pi * x) * sin(pi * y)

function jacobi!(u, u_new, rhs, h, tol, maxiter)
    N = size(u, 1)
    update_error = Inf
    iterations = 0

    for iter in 1:maxiter
        update_error = 0.0

        # Interior points only.
        # Boundary points remain zero (Dirichlet).
        for j in 2:N-1
            for i in 2:N-1
                u_new[i, j] = 0.25 * (
                    u[i+1, j] +
                    u[i-1, j] +
                    u[i, j+1] +
                    u[i, j-1] +
                    h^2 * rhs[i, j]
                )

                update_error =
                    max(update_error, abs(u_new[i, j] - u[i, j]))
            end
        end

        u .= u_new
        iterations = iter

        if iter % 1000 == 0
            @printf(
                "iteration = %6d, update error = %.6e\n",
                iter,
                update_error
            )
        end

        if update_error < tol
            break
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
    for j in 1:N
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

    out = joinpath(@__DIR__, "poisson_jacobi.png")
    save_plot(out, x, y, u, ue, err)
    println("saved ", out)

    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
