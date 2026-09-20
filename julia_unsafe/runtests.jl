using Test
using Random
include("poisson.jl")

# Independent, ordinary two-buffer implementation in the original order.
function reference!(u, v, rhs, h, tol, maxiter)
    n = size(u, 1)
    error = Inf
    iterations = 0
    for iter in 1:maxiter
        report = iter % REPORT_INTERVAL == 0 || iter == maxiter
        error_now = 0.0
        for j in 2:n-1, i in 2:n-1
            value = 0.25 * (u[i, j+1] + u[i, j-1] + u[i+1, j] + u[i-1, j] +
                            (h * h) * rhs[i, j])
            error_now = max(error_now, abs(value - u[i, j]))
            v[i, j] = value
        end
        u, v = v, u
        iterations = iter
        if report
            error = error_now
            error < tol && break
        end
    end
    return u, iterations, error
end

@testset "Temporally blocked Jacobi" begin
    rng = MersenneTwister(1234)
    redirect_stdout(devnull) do
        # Small grids exercise the wavefront's overlapping start/end regions;
        # odd sizes exercise SIMD tails, and counts straddle reporting points.
        for n in (3:36..., 401)
            counts = n == 401 ? (9, 18) : (0, 1, 2, 7, 8, 9, 10, 17, 999, 1000, 1001, 2003)
            for count in counts
                u = randn(rng, n, n)
                v = copy(u) # Identical fixed (including nonzero) boundaries.
                v[2:end-1, 2:end-1] .= randn(rng, n-2, n-2)
                rhs = randn(rng, n, n)
                rhs_before = copy(rhs)
                expected = reference!(copy(u), copy(v), rhs, 0.13, 0.0, count)
                actual = jacobi!(u, v, rhs, 0.13, 0.0, count)
                @test reinterpret(UInt64, vec(actual[1])) == reinterpret(UInt64, vec(expected[1]))
                @test actual[2:3] == expected[2:3]
                @test rhs == rhs_before
            end
        end
        for count in (17, 1000, 1001, 2003)
            actual = jacobi!(zeros(5,5), zeros(5,5), zeros(5,5), 0.25, 1e-10, count)
            @test actual[2] == min(count, REPORT_INTERVAL)
            @test actual[3] == 0.0
        end
        u = zeros(5,5)
        @test_throws ArgumentError jacobi!(u, u, copy(u), 0.25, 1e-10, 10)
        @test_throws ArgumentError jacobi!(u, copy(u), u, 0.25, 1e-10, 10)
        @test_throws ArgumentError jacobi!(u, zeros(6,6), copy(u), 0.25, 1e-10, 10)
    end
end
