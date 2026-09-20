# Temporally blocked Jacobi

The solver keeps the original single-threaded Jacobi method, stencil addition
order, Float64 arithmetic, fixed boundaries, and convergence checks. It advances
up to eight sweeps along a column wavefront through the same two solution buffers.
This reuses nearby columns in cache before moving on. The products `h² * rhs` are
computed once per solve; their Float64 rounding is unchanged. No fast-math,
explicit FMA, additional solver dependency, or problem-specific shortcut is used.

For diagonal `d`, stage `t` computes column `j = d - t`. The preceding stage has
already computed column `j + 1`, making every input available. The stage two steps
behind can be overwritten because its last consumer has finished. After an even
number of stages, `u` holds the latest iterate. Reporting sweeps remain separate,
so the update width compares consecutive iterates at exactly the original
reporting points, including an odd final iteration.

The public `jacobi!` entry point checks that the three inputs are distinct,
equally sized square `Matrix{Float64}` buffers with at least three grid points.
Both solution buffers must contain the desired fixed boundary values; `main`
initializes them to zero. Raw-pointer kernels preserve their owning arrays with
`GC.@preserve`.

## Verification and timing

From the repository root:

```sh
julia --project=julia_unsafe julia_unsafe/runtests.jl
julia --project=julia_unsafe julia_unsafe/benchmark.jl 5
julia --project=julia_unsafe julia_unsafe/poisson.jl
```

Tests compare with an independent, ordinary two-buffer implementation using the
original arithmetic order, including random inputs, nonzero fixed boundaries,
small grids, SIMD tails, odd iteration counts, early stopping, and reporting
boundaries. The comparison is bitwise for the solution, and exact for the iteration
count and final update width.

The benchmark builds the unchanged C++ implementation with `cxx/build.sh` and
alternates execution order. Both solvers use N=401, tolerance 1e-10, and 100,000
iterations. It checks the final reported error as well as the iteration count.
C++ figures are written into a temporary directory.

`main` and the benchmark warm up Julia's solver on a 3×3 problem before timing.
Reported times exclude compilation, input-grid initialization, and plotting;
they include right-hand-side scaling inside `jacobi!`, convergence checks, and
progress output (redirected to `/dev/null` in the Julia benchmark). C++ timing
likewise excludes compilation and plotting. These are solver times, not process
startup or end-to-end times. Host load and thermal state can affect small timing
differences; use repeated measurements on the target machine.

On this Apple arm64 host with Julia 1.13.0, five alternating runs produced:

| Round | Julia (s) | C++ (s) |
|---|---:|---:|
| 1 | 3.654119 | 3.705212 |
| 2 | 3.634631 | 3.710242 |
| 3 | 3.631475 | 3.722405 |
| 4 | 3.642505 | 3.726362 |
| 5 | 3.860469 | 3.749110 |
| Median | 3.642505 | 3.722405 |

The median solver time was 2.1% lower (1.022× speedup). One of five Julia runs
was slower than C++, so this is a small measured improvement, not a guarantee
that Julia wins every run. The original Julia solver took about 4.00 s in an
initial, compilation-inclusive run; that figure is not a matched warm benchmark.
