# Problem & API

## Problem Definition

Use `TVProblem` to define a denoising/reconstruction task:

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

Key inputs:

- `f`: `AbstractArray{<:AbstractFloat,N}` data.
- `lambda`: TV weight (`>= 0`).
- `spacing`: `nothing`, scalar, `NTuple{N}`, or length-`N` vector.
- `data_fidelity`: `L2Fidelity()` or `PoissonFidelity()`.
- `tv_mode`: `IsotropicTV()` or `AnisotropicTV()`.
- `boundary`: currently `Neumann()`.

`TVProblem` is immutable, but it stores `f` by reference. For repeated solves
with the same shape/eltype, update `f` in place (for example `copyto!`) and
reuse the same `TVProblem`.

## Solving a Single Input

Allocate output internally:

```julia
u, stats = TVImageFiltering.solve(problem, TVImageFiltering.ROFConfig())
```

Or in place:

```julia
u = copy(problem.f)
stats = TVImageFiltering.solve!(u, problem, TVImageFiltering.ROFConfig())
```

`stats` is:

```julia
TVImageFiltering.SolverStats(iterations, converged, rel_change)
```

Supported solvers:

- `ROFConfig` (ROF model with `L2Fidelity`).
- `PDHGConfig` (`L2Fidelity` and `PoissonFidelity`).

## Warm Starts and State Reuse

For repeated solves, pass reusable workspace:

```julia
state = TVImageFiltering.ROFState(problem.f)   # or PDHGState(problem.f)
u = copy(problem.f)
stats = TVImageFiltering.solve!(u, problem, TVImageFiltering.ROFConfig(); state = state)
```

Behavior:

- `u` provides primal warm start.
- `state.p` (dual variable) is retained across calls.
- State shape/eltype must match the solve buffer.

## Batch API

For arrays shaped `(spatial..., batch)`:

```julia
u_batch, stats = TVImageFiltering.solve_batch(
    f_batch,
    TVImageFiltering.PDHGConfig();
    lambda = 0.1,
    data_fidelity = TVImageFiltering.L2Fidelity(),
)
```

TV operators act on spatial axes only; the last axis is batch index.

The batch `stats` aggregate per-sample results:

- `iterations`: max iterations across samples.
- `converged`: true only if all samples converged.
- `rel_change`: max relative change across samples.
