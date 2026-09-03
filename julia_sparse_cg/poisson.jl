using IterativeSolvers
using LinearAlgebra
using Plots
using Printf
using SparseArrays

# -----------------------------------------------------------------------------
# Solve
#
#   -Δu = f  in (0, 1) × (0, 1),    u = 0 on the boundary,
#
# with the exact solution u(x, y) = sin(πx)sin(πy).  Only the interior
# grid points are unknowns in the sparse linear system.
# -----------------------------------------------------------------------------

u_exact(x, y) = sinpi(x) * sinpi(y)
forcing(x, y) = 2pi^2 * sinpi(x) * sinpi(y)

"""Build the CSC matrix for the five-point discretization of `-Δ`."""
function poisson_matrix(interior_size, h)
    scale = inv(h^2)
    diagonal = fill(2scale, interior_size)
    off_diagonal = fill(-scale, interior_size - 1)
    one_dimensional = spdiagm(
        -1 => off_diagonal,
         0 => diagonal,
         1 => off_diagonal,
    )
    identity = spdiagm(0 => ones(interior_size))
    return kron(identity, one_dimensional) + kron(one_dimensional, identity)
end

function save_plot(path, coordinates, u, exact, error)
    exact_plot = surface(
        coordinates,
        coordinates,
        exact';
        xlabel="x",
        ylabel="y",
        zlabel="u",
        title="Exact solution",
        camera=(45, 30),
    )

    numerical_plot = surface(
        coordinates,
        coordinates,
        u';
        xlabel="x",
        ylabel="y",
        zlabel="u",
        title="Numerical solution",
        camera=(45, 30),
    )

    error_plot = heatmap(
        coordinates,
        coordinates,
        error';
        xlabel="x",
        ylabel="y",
        title="Absolute error",
        colorbar_title="|u - u_exact|",
    )

    figure = plot(
        exact_plot,
        numerical_plot,
        error_plot;
        layout=(1, 3),
        size=(1500, 450),
    )
    savefig(figure, path)
    return nothing
end

function main()
    grid_size = 101
    relative_tolerance = 1e-10
    maximum_iterations = 100_000

    coordinates = range(0.0, 1.0, length=grid_size)
    h = inv(grid_size - 1)
    interior_size = grid_size - 2

    @printf("N = %d\n", grid_size)
    @printf("h = %.6e\n", h)

    operator = poisson_matrix(interior_size, h)
    rhs = [
        forcing(coordinates[i], coordinates[j])
        for i in 2:grid_size-1, j in 2:grid_size-1
    ]
    @printf("unknowns = %d\n", size(operator, 1))
    @printf("nnz(A) = %d\n", nnz(operator))

    # This particular right-hand side is an eigenvector of the discrete
    # five-point Laplacian, so exact-arithmetic CG spans the solution after one
    # iteration. A general Poisson right-hand side will require more steps.
    solution_vector = Vector{Float64}()
    history = nothing
    duration = @elapsed begin
        solution_vector, history = cg(
            operator,
            vec(rhs);
            reltol=relative_tolerance,
            maxiter=maximum_iterations,
            log=true,
        )
    end

    solution = zeros(grid_size, grid_size)
    solution[2:grid_size-1, 2:grid_size-1] .=
        reshape(solution_vector, interior_size, interior_size)

    exact = [u_exact(x, y) for x in coordinates, y in coordinates]
    error = abs.(solution .- exact)
    max_error = maximum(error)
    l2_error = sqrt(sum(abs2, solution .- exact) * h^2)
    relative_residual = norm(operator * solution_vector - vec(rhs)) / norm(rhs)

    println()
    @printf("time = %.6f seconds\n", duration)
    @printf("CG iterations = %d\n", history.iters)
    @printf("relative residual = %.6e\n", relative_residual)
    println()
    @printf("max error = %.6e\n", max_error)
    @printf("L2 error  = %.6e\n", l2_error / sqrt(grid_size))

    output_path = joinpath(@__DIR__, "poisson_sparse_cg.png")
    save_plot(output_path, coordinates, solution, exact, error)
    println("saved ", output_path)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
