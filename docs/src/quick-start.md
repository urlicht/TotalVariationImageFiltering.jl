# Quick Start

```julia
using Random
using Statistics
using TVImageFiltering

Random.seed!(42)

# Synthetic piecewise-constant image
n, m = 128, 128
clean = zeros(Float32, n, m)
clean[33:96, 33:96] .= 1f0

# Add Gaussian noise
noisy = clean .+ 0.15f0 * randn(Float32, n, m)

problem = TVImageFiltering.TVProblem(
    noisy;
    lambda = 0.12f0,
    tv_mode = TVImageFiltering.IsotropicTV(),
    spacing = (1.0f0, 1.0f0),
)

config = TVImageFiltering.ROFConfig(
    maxiter = 1000,
    tau = 0.0625f0,
    tol = 1f-5,
    check_every = 10,
)

denoised, stats = TVImageFiltering.solve(problem, config)

println("Converged: ", stats.converged)
println("Iterations: ", stats.iterations)
println("Relative change: ", stats.rel_change)
println("Noisy MSE: ", mean(abs2, noisy .- clean))
println("Denoised MSE: ", mean(abs2, denoised .- clean))
```

## Next Steps

- See [Problem & API](problem-and-api.md) for all constructor/solver options.
- See [ROF Solver](rof-solver.md) and [PDHG Solver](pdhg-solver.md) for math and stability constraints.
- See [Lambda Selection](lambda-selection.md) for automatic regularization tuning.
