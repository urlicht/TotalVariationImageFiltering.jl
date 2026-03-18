"""
TVImageFiltering.jl provides total-variation denoising primitives for
N-dimensional arrays.

Main entry points:
- `TVProblem(...)` to describe a denoising problem
- `solve(problem, config)` / `solve!(u, problem, config)`
- `solve_batch(...)` / `solve_batch!(...)`
"""
module TVImageFiltering

include("types.jl")
include("problem.jl")
include("operators/operators.jl")
include("solvers/solvers.jl")
include("api.jl")
include("lambda_selection.jl")

end # module TVImageFiltering
