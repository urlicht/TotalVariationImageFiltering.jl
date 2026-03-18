# Batch & CUDA

## Batch API (CPU and Generic Arrays)

Use `solve_batch` for input arrays shaped:

```text
(spatial_dim_1, spatial_dim_2, ..., spatial_dim_k, batch)
```

The last axis is treated as batch index, and TV operators are applied only
across spatial axes.

Example:

```julia
u_batch, stats = TVImageFiltering.solve_batch(
    f_batch,
    TVImageFiltering.ROFConfig();
    lambda = 0.1,
    tv_mode = TVImageFiltering.IsotropicTV(),
)
```

For PDHG batch solves, you can additionally pass:

- `constraint = TVImageFiltering.NonnegativeConstraint()`, or
- `constraint = TVImageFiltering.BoxConstraint(lower, upper)`.

Batch state reuse:

- pass `state = [ROFState(slice1), ROFState(slice2), ...]` for ROF;
- pass `state = [PDHGState(slice1), PDHGState(slice2), ...]` for PDHG.

State vector length must match batch size.

## CUDA Extension

The extension module `TVImageFilteringCUDAExt` is loaded automatically when:

- `CUDA.jl` is installed and loaded,
- a functional CUDA runtime/device is available.

Example:

```julia
using CUDA
using TVImageFiltering

f_gpu = CUDA.rand(Float32, 256, 256)
problem_gpu = TVImageFiltering.TVProblem(f_gpu; lambda = 0.15f0)
u_gpu, stats_gpu = TVImageFiltering.solve(problem_gpu, TVImageFiltering.ROFConfig())
```

## CUDA Coverage

Current behavior based on extension code/tests:

- CUDA kernels are provided for gradient/divergence/projection primitives.
- Single-image ROF and PDHG on `CuArray` are supported.
- Batched CUDA solve is specialized for `ROFConfig` and `PDHGConfig`.
- Batched CUDA path currently requires:
  - `L2Fidelity` (ROF), or `L2Fidelity` / `PoissonFidelity` (PDHG),
  - `Neumann` boundary.

ROF paths currently support only `constraint = NoConstraint()`.

If CUDA is unavailable, CPU paths continue to work.

## Numerical Equivalence Checks

Repository tests compare CPU and CUDA outputs with tolerances for:

- single-image ROF solve;
- batched ROF solve.
