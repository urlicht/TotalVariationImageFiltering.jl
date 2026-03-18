"""
Solve a TV problem and return `(u, stats)`.

`config` selects the solver backend. Implemented: `ROFConfig`, `PDHGConfig`.
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

"""
Solve a batch of same-size images stored in one array.

The last axis is interpreted as batch index, and TV operators act only on the
leading spatial axes.
"""
function solve_batch(
    f_batch::AbstractArray{T,N},
    config::AbstractTVSolver = ROFConfig();
    lambda::Real,
    spacing = nothing,
    data_fidelity::AbstractDataFidelity = L2Fidelity(),
    tv_mode::AbstractTVMode = IsotropicTV(),
    boundary::AbstractBoundaryCondition = Neumann(),
    constraint::AbstractPrimalConstraint = NoConstraint(),
    init::Union{Nothing,AbstractArray} = nothing,
    state = nothing,
) where {T<:AbstractFloat,N}
    N >= 2 ||
        throw(ArgumentError("f_batch must have at least 2 dimensions (spatial..., batch)"))

    u_batch = init === nothing ? copy(f_batch) : copy(init)
    size(u_batch) == size(f_batch) || throw(ArgumentError("init must match f_batch size"))

    stats = solve_batch!(
        u_batch,
        f_batch,
        config;
        lambda = lambda,
        spacing = spacing,
        data_fidelity = data_fidelity,
        tv_mode = tv_mode,
        boundary = boundary,
        constraint = constraint,
        state = state,
    )
    return u_batch, stats
end

function solve_batch!(
    u_batch::AbstractArray,
    f_batch::AbstractArray,
    config::AbstractTVSolver;
    kwargs...,
)
    throw(MethodError(solve_batch!, (u_batch, f_batch, config)))
end

function solve_batch!(
    u_batch::AbstractArray{T,N},
    f_batch::AbstractArray{T,N},
    config::ROFConfig;
    lambda::Real,
    spacing = nothing,
    data_fidelity::AbstractDataFidelity = L2Fidelity(),
    tv_mode::AbstractTVMode = IsotropicTV(),
    boundary::AbstractBoundaryCondition = Neumann(),
    constraint::AbstractPrimalConstraint = NoConstraint(),
    state = nothing,
) where {T<:AbstractFloat,N}
    N >= 2 ||
        throw(ArgumentError("f_batch must have at least 2 dimensions (spatial..., batch)"))
    size(u_batch) == size(f_batch) ||
        throw(ArgumentError("u_batch and f_batch must have matching sizes"))
    constraint isa NoConstraint || throw(
        ArgumentError(
            "ROF currently supports only unconstrained problems; set constraint = NoConstraint() or use PDHGConfig",
        ),
    )

    batch_count = size(f_batch, N)
    local_states = if state === nothing
        nothing
    elseif state isa AbstractVector
        length(state) == batch_count || throw(
            ArgumentError("state vector length must equal batch size $batch_count"),
        )
        state
    else
        throw(ArgumentError("state must be `nothing` or a vector of per-image state objects"))
    end

    max_iterations = 0
    converged = true
    max_rel_change = zero(T)

    @views for b = 1:batch_count
        f_view = selectdim(f_batch, N, b)
        u_view = selectdim(u_batch, N, b)
        problem = TVProblem(
            f_view;
            lambda = lambda,
            spacing = spacing,
            data_fidelity = data_fidelity,
            tv_mode = tv_mode,
            boundary = boundary,
            constraint = constraint,
        )

        stats =
            local_states === nothing ? solve!(u_view, problem, config) :
            solve!(u_view, problem, config; state = local_states[b])

        max_iterations = max(max_iterations, stats.iterations)
        converged &= stats.converged
        max_rel_change = max(max_rel_change, stats.rel_change)
    end

    return SolverStats{T}(max_iterations, converged, max_rel_change)
end

function solve_batch!(
    u_batch::AbstractArray{T,N},
    f_batch::AbstractArray{T,N},
    config::PDHGConfig;
    lambda::Real,
    spacing = nothing,
    data_fidelity::AbstractDataFidelity = L2Fidelity(),
    tv_mode::AbstractTVMode = IsotropicTV(),
    boundary::AbstractBoundaryCondition = Neumann(),
    constraint::AbstractPrimalConstraint = NoConstraint(),
    state = nothing,
) where {T<:AbstractFloat,N}
    N >= 2 ||
        throw(ArgumentError("f_batch must have at least 2 dimensions (spatial..., batch)"))
    size(u_batch) == size(f_batch) ||
        throw(ArgumentError("u_batch and f_batch must have matching sizes"))

    batch_count = size(f_batch, N)
    local_states = if state === nothing
        nothing
    elseif state isa AbstractVector
        length(state) == batch_count || throw(
            ArgumentError("state vector length must equal batch size $batch_count"),
        )
        state
    else
        throw(ArgumentError("state must be `nothing` or a vector of per-image state objects"))
    end

    max_iterations = 0
    converged = true
    max_rel_change = zero(T)

    @views for b = 1:batch_count
        f_view = selectdim(f_batch, N, b)
        u_view = selectdim(u_batch, N, b)
        problem = TVProblem(
            f_view;
            lambda = lambda,
            spacing = spacing,
            data_fidelity = data_fidelity,
            tv_mode = tv_mode,
            boundary = boundary,
            constraint = constraint,
        )

        stats =
            local_states === nothing ? solve!(u_view, problem, config) :
            solve!(u_view, problem, config; state = local_states[b])

        max_iterations = max(max_iterations, stats.iterations)
        converged &= stats.converged
        max_rel_change = max(max_rel_change, stats.rel_change)
    end

    return SolverStats{T}(max_iterations, converged, max_rel_change)
end
