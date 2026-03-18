# TVImageFiltering.jl

`TVImageFiltering.jl` provides total-variation denoising tools for N-dimensional arrays.

## Documentation

- [ROF Solver](rof-solver.md): mathematical formulation and the exact iterative process used by this package.

## Quick Start

```julia
using TVImageFiltering

f = rand(Float32, 128, 128)
problem = TVImageFiltering.TVProblem(f; lambda = 0.1f0)
u, stats = TVImageFiltering.solve(problem, TVImageFiltering.ROFConfig())
```
