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
#
# Single-threaded Jacobi with column-wise temporal blocking. Eight sweeps
# advance as a wavefront through two buffers, reusing recently accessed
# columns in cache. The arithmetic order matches the original pointer
# implementation; no fast-math or fused multiply-add is used.
# Only interior points are written. Update widths are checked every 1000
# iterations and at maxiter, just as in the ordinary two-buffer solver.
# ------------------------------------------------------------

u_exact(x, y) = sin(pi * x) * sin(pi * y)
f(x, y) = 2pi^2 * sin(pi * x) * sin(pi * y)

const REPORT_INTERVAL = 1000
const TEMPORAL_STEPS = 8

"""
One Jacobi sweep over the interior using the five-point stencil.

`TRACK` is a compile-time flag: when it is `false` the update-width
reduction is compiled away, so the inner loop stays reduction-free.
Only every `REPORT_INTERVAL`-th sweep tracks the error (see `jacobi!`).
"""
function sweep!(u, u_new, rhs, h2, n, ::Val{TRACK}) where {TRACK}
    error = 0.0
    GC.@preserve u u_new rhs begin
        pu = pointer(u)
        pu_new = pointer(u_new)
        prhs = pointer(rhs)

        # Interior points only.
        # Boundary points remain zero (Dirichlet).
        @inbounds for j in 2:n-1
            base = (j - 1) * n
            @simd for i in 2:n-1
                k = base + i
                value = 0.25 * (unsafe_load(pu, k + n) + unsafe_load(pu, k - n) +
                                unsafe_load(pu, k + 1) + unsafe_load(pu, k - 1) +
                                h2 * unsafe_load(prhs, k))
                if TRACK
                    error = max(error, abs(value - unsafe_load(pu, k)))
                end
                unsafe_store!(pu_new, value, k)
            end
        end
    end

    return error
end

"""
Advance an even number of sweeps, leaving the latest iterate in `u`.

At diagonal `d`, stage `t` updates column `j = d - t`. Stage `t - 1`
has already produced column `j + 1`, so all four neighbors are ready.
Overwriting stage `t - 2` is safe: stage `t - 1` has consumed its last
needed neighbor. The untouched boundary columns serve every stage.
`scaled_rhs` contains the once-rounded products `h² * rhs`.
"""
function sweep_batch!(u, u_new, scaled_rhs, n, steps)
    GC.@preserve u u_new scaled_rhs begin
        pu, pv, pr = pointer(u), pointer(u_new), pointer(scaled_rhs)
        for d in 3:n-1+steps
            for t in max(1, d-(n-1)):min(steps, d-2)
                j = d - t
                src, dst = isodd(t) ? (pu, pv) : (pv, pu)
                base = (j - 1) * n
                @simd ivdep for i in 2:n-1
                    k = base + i
                    value = 0.25 * (unsafe_load(src, k + n) + unsafe_load(src, k - n) +
                                    unsafe_load(src, k + 1) + unsafe_load(src, k - 1) +
                                    unsafe_load(pr, k))
                    unsafe_store!(dst, value, k)
                end
            end
        end
    end
    return nothing
end

function jacobi!(u::Matrix{Float64}, u_new::Matrix{Float64}, rhs::Matrix{Float64},
                 h, tol, maxiter)
    n = size(u, 1)
    size(u) == size(u_new) == size(rhs) == (n, n) && n >= 3 ||
        throw(ArgumentError("expected equally sized square grids with n >= 3"))
    (Base.mightalias(u, u_new) || Base.mightalias(u, rhs) || Base.mightalias(u_new, rhs)) &&
        throw(ArgumentError("Jacobi buffers and rhs must not alias"))
    h2 = h * h
    scaled_rhs = h2 .* rhs
    update_error = Inf
    iterations = 0

    while iterations < maxiter
        next_report = min((iterations ÷ REPORT_INTERVAL + 1) * REPORT_INTERVAL, maxiter)
        # Leave the reporting sweep separate to measure consecutive iterates.
        steps = min(TEMPORAL_STEPS, next_report - iterations - 1)
        steps -= isodd(steps)
        if steps >= 2
            sweep_batch!(u, u_new, scaled_rhs, n, steps)
            iterations += steps
            continue
        end

        iterations += 1
        report = iterations == next_report
        if report
            update_error = sweep!(u, u_new, rhs, h2, n, Val(true))
        else
            sweep!(u, u_new, rhs, h2, n, Val(false))
        end
        u, u_new = u_new, u

        if report
            @printf("iteration = %6d, update error = %.6e\n", iterations, update_error)
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
        xlabel = "x",
        ylabel = "y",
        zlabel = "u",
        title = "Exact solution",
        camera = (45, 30)
    )

    p2 = surface(
        x,
        y,
        u',
        xlabel = "x",
        ylabel = "y",
        zlabel = "u",
        title = "Numerical solution",
        camera = (45, 30)
    )

    p3 = heatmap(
        x,
        y,
        err',
        xlabel = "x",
        ylabel = "y",
        title = "Absolute error",
        colorbar_title = "|u - u_exact|"
    )

    p = plot(p1, p2, p3, layout = (1, 3), size = (1500, 450))

    display(p)
    savefig(p, path)
    return p
end

function main()
    N = 401
    tol = 1e-10
    maxiter = 100_000

    x = range(0.0, 1.0, length = N)
    y = range(0.0, 1.0, length = N)
    h = 1.0 / (N - 1)

    @printf("N = %d\n", N)
    @printf("h = %.6e\n", h)

    u = zeros(Float64, N, N)
    u_new = zeros(Float64, N, N)
    rhs = [f(x[i], y[j]) for i in 1:N, j in 1:N]

    # Compile the solver on a tiny problem before timing, matching the C++
    # solve-only measurement (which also excludes compilation).
    redirect_stdout(devnull) do
        jacobi!(zeros(3, 3), zeros(3, 3), zeros(3, 3), 1.0, 0.0, 10)
    end
    start_time = time()
    u, iterations, update_error = jacobi!(u, u_new, rhs, h, tol, maxiter)
    duration = time() - start_time

    println()
    @printf("time = %.6f seconds\n", duration)
    @printf("Jacobi iterations = %d\n", iterations)
    @printf("final update error = %.6e\n", update_error)

    ue = [u_exact(x[i], y[j]) for i in 1:N, j in 1:N]
    err = abs.(u .- ue)
    max_error = maximum(err)
    l2_error = sqrt(sum((u .- ue) .^ 2) * h^2)

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
