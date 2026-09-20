# Run from any directory: julia --project=julia_unsafe julia_unsafe/benchmark.jl [repeats]
using Statistics
include("poisson.jl")

function benchmark(repeats)
    repeats > 0 || throw(ArgumentError("repeats must be positive"))
    cxx_dir = normpath(joinpath(@__DIR__, "..", "cxx"))
    run(`bash $(joinpath(cxx_dir, "build.sh"))`)
    n = 401
    h = 1.0 / (n - 1)
    x = range(0.0, 1.0, length=n)
    rhs = [f(x[i], x[j]) for i in 1:n, j in 1:n]
    exact = [u_exact(x[i], x[j]) for i in 1:n, j in 1:n]
    redirect_stdout(devnull) do
        jacobi!(zeros(3,3), zeros(3,3), zeros(3,3), 1.0, 0.0, 10)
    end
    julia_times, cxx_times = Float64[], Float64[]
    mktempdir() do output_dir
        for rep in 1:repeats
            # Alternate ordering to reduce systematic thermal/order bias.
            for implementation in (isodd(rep) ? (:julia, :cxx) : (:cxx, :julia))
                if implementation == :julia
                    u, v = zeros(n,n), zeros(n,n)
                    elapsed, result = redirect_stdout(devnull) do
                        start = time_ns()
                        result = jacobi!(u, v, rhs, h, 1e-10, 100_000)
                        ((time_ns() - start) / 1e9, result)
                    end
                    @assert result[2] == 100_000
                    @assert isapprox(maximum(abs.(result[1] .- exact)), 4.575793e-2; atol=5e-9)
                    @assert isapprox(result[3], 1.411484e-6; atol=5e-13)
                    push!(julia_times, elapsed)
                    @printf("round %d Julia: %.6f s\n", rep, elapsed)
                else
                    output = read(Cmd(`$(joinpath(cxx_dir, "poisson"))`; dir=output_dir), String)
                    elapsed = parse(Float64, match(r"time = ([0-9.]+) seconds", output)[1])
                    @assert occursin("Jacobi iterations = 100000", output)
                    @assert occursin("max error = 4.575793e-02", output)
                    @assert occursin("final update error = 1.411484e-06", output)
                    push!(cxx_times, elapsed)
                    @printf("round %d C++:   %.6f s\n", rep, elapsed)
                end
            end
        end
    end
    @printf("median Julia: %.6f s; C++: %.6f s; speedup: %.3fx\n",
            median(julia_times), median(cxx_times), median(cxx_times) / median(julia_times))
    @printf("minimum Julia: %.6f s; C++: %.6f s\n", minimum(julia_times), minimum(cxx_times))
end

benchmark(isempty(ARGS) ? 5 : parse(Int, only(ARGS)))
