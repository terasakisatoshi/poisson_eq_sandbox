# Temporally blocked Jacobi

The solver keeps the original single-threaded Jacobi method, stencil addition
order, Float64 arithmetic, fixed boundaries, and convergence checks. It advances
up to eight sweeps along a column wavefront through the same two solution buffers.
Pairs of stages also share intermediate SIMD values in registers, using
16-point blocks and 8/4/2/1-point tails. This reuses nearby columns in cache
before moving on. The products `h² * rhs` are
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
julia --project=julia_unsafe julia_unsafe/inspect_native.jl /tmp/poisson-native
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

The machine-code comparison and latest measurements are recorded in
[native_analysis.md](native_analysis.md). `inspect_native.jl` uses `@code_native`
with instruction encodings and disassembles the unchanged C++ executable.
It also records the compiler/CPU versions and vector-load LLVM IR.

On an Apple M4 with Julia 1.13.0 and Apple clang 21.0.0, five runs in rotating
order produced the following solver times. “Before” is the eight-stage wavefront
implementation before SIMD stage fusion; all Julia measurements were warmed up.

| Round | Julia before (s) | Julia after (s) | C++ (s) |
|---|---:|---:|---:|
| 1 | 4.699281 | 3.385435 | 3.846381 |
| 2 | 3.637510 | 3.504544 | 3.913423 |
| 3 | 4.079874 | 3.818548 | 3.695129 |
| 4 | 3.846951 | 3.413451 | 3.707956 |
| 5 | 3.633087 | 3.517270 | 3.708979 |
| Median | 3.846951 | 3.504544 | 3.708979 |
| Minimum | 3.633087 | 3.385435 | 3.695129 |

The new median was 8.9% lower than the previous Julia implementation and 5.5%
lower than C++. One run was slower than C++, and the first baseline measurement
was an outlier, so these timings should not be treated as universal speedups.
All five full-grid comparisons with the previous Julia implementation were
bitwise identical, including the final update width and iteration count.
The unit suite passes 1,241 checks covering every SIMD remainder on grids 3–36
as well as the 401-point grid.
