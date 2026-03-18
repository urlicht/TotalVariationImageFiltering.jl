# TVImageFiltering.jl

`TVImageFiltering.jl` is a Julia package for total-variation (TV) image denoising and reconstruction.

It currently implements the ROF model with Chambolle's dual projection method:

```math
\min_u \frac{1}{2}\|u-f\|_2^2 + \lambda \, \mathrm{TV}(u)
```

The package supports:
- N-dimensional `AbstractArray{T,N}` inputs (`T <: AbstractFloat`)
- Isotropic and anisotropic TV
- Configurable grid spacing
- Single-image and batched solves
- Optional CUDA acceleration through package extension

## Installation

If you are working from this local repository:

```julia
julia --project=.
```

To use this package from another Julia environment, `dev` the local path:

```julia
import Pkg
Pkg.develop(path="/absolute/path/to/TVImageFiltering.jl")
```

If the repository is hosted remotely, you can also `add` by URL:

```julia
import Pkg
Pkg.add(url="https://github.com/<owner>/TVImageFiltering.jl")
```

## Quick Example

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

## API Overview

### Problem definition

```julia
problem = TVImageFiltering.TVProblem(
    f;
    lambda,
    spacing = nothing,
    data_fidelity = TVImageFiltering.L2Fidelity(),
    tv_mode = TVImageFiltering.IsotropicTV(),
    boundary = TVImageFiltering.Neumann(),
)
```

- `f`: input array (`AbstractArray{<:AbstractFloat,N}`)
- `lambda`: TV weight (`>= 0`)
- `spacing`: `nothing`, scalar, `NTuple{N}`, or length-`N` vector
- `tv_mode`: `IsotropicTV()` or `AnisotropicTV()`

### Solving one image

```julia
u, stats = TVImageFiltering.solve(problem, TVImageFiltering.ROFConfig(); init = nothing)
```

`stats` is `SolverStats(iterations, converged, rel_change)`.

### Solving in-place

```julia
stats = TVImageFiltering.solve!(u, problem, TVImageFiltering.ROFConfig(); state = nothing)
```

Use `state = TVImageFiltering.ROFState(reference_array)` to reuse buffers across repeated calls.

### Solving a batch

```julia
u_batch, stats = TVImageFiltering.solve_batch(
    f_batch,
    TVImageFiltering.ROFConfig();
    lambda = 0.1,
    spacing = nothing,
    tv_mode = TVImageFiltering.IsotropicTV(),
)
```

For `f_batch` with shape `(spatial..., batch)`, TV operators are applied only over spatial dimensions and the last dimension is treated as batch index.

## Optional CUDA Usage

The CUDA extension loads automatically when both `TVImageFiltering` and `CUDA` are available.

```julia
using CUDA
using TVImageFiltering

f_gpu = CUDA.rand(Float32, 256, 256)
problem_gpu = TVImageFiltering.TVProblem(f_gpu; lambda = 0.15f0)
u_gpu, stats_gpu = TVImageFiltering.solve(problem_gpu, TVImageFiltering.ROFConfig())

# Move back to CPU if needed
u_cpu = Array(u_gpu)
```

Batch solving on GPU is also supported via `solve_batch` with `CuArray` input.

## Current Scope and Constraints

- Implemented solver: `ROFConfig` (ROF denoising model)
- Supported data fidelity in solver: `L2Fidelity`
- Boundary condition implementation: `Neumann`
- `tau` must satisfy the stability bound:
  - `tau < 1 / (2 * sum(h_d^(-2)))` over spatial dimensions with size > 1
  - with unit spacing in 2D, this means `tau < 0.25`

## Running Tests

```julia
import Pkg
Pkg.test()
```

CUDA tests run only when CUDA is installed and functional.
