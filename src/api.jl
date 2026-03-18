"""
Solve a TV problem and return `(u, stats)`.

`config` selects the solver backend (`ROFConfig`, `PDHGConfig`, `ADMMConfig`).
"""
function solve(
    problem::TVProblem,
    config::AbstractTVSolver = ROFConfig();
    init::Union{Nothing,AbstractArray} = nothing,
    kwargs...,
)
    u = init === nothing ? copy(problem.f) : copy(init)
    size(u) == size(problem.f) || throw(ArgumentError("init must match problem.f size"))
    stats = solve!(u, problem, config; kwargs...)
    return u, stats
end

function solve!(u::AbstractArray, problem::TVProblem, config::AbstractTVSolver; kwargs...)
    throw(MethodError(solve!, (u, problem, config)))
end
