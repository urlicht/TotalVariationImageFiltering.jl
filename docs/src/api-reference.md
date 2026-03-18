# API Reference

```@meta
CurrentModule = TVImageFiltering
```

`TVImageFiltering.jl` does not currently export symbols, so call APIs with the
`TVImageFiltering.` prefix in user code.

## Module

```@docs
TVImageFiltering
```

## Types

```@docs
AbstractBoundaryCondition
Neumann
AbstractDataFidelity
L2Fidelity
PoissonFidelity
AbstractTVMode
IsotropicTV
AnisotropicTV
AbstractTVSolver
TVProblem
SolverStats
ROFConfig
ROFState
PDHGConfig
PDHGState
DiscrepancySelection
SURESelection
```

## Solvers

```@docs
solve
solve!
solve_batch
```

## Lambda Selection

```@docs
select_lambda_discrepancy
select_lambda_sure
```

## Operator Utilities

```@docs
allocate_dual
gradient!
divergence!
project_dual_ball!
```
