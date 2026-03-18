struct ROFConfig{T<:AbstractFloat} <: AbstractTVSolver
    maxiter::Int
    tau::T
    tol::T
    check_every::Int
end

function ROFConfig(;
    maxiter::Int = 300,
    tau::Real = 0.0625, # τ = 0.25 max for 2D, see 2nd remark after proof of Theorem 3.1.
    tol::Real = 1e-4,
    check_every::Int = 10,
)
    T = promote_type(typeof(float(tau)), typeof(float(tol)))
    return ROFConfig{T}(maxiter, T(tau), T(tol), check_every)
end

function _validate(config::ROFConfig)
    _validate_common_config(config.maxiter, config.check_every)
    config.tau > zero(config.tau) || throw(ArgumentError("tau must be positive"))
    config.tol >= zero(config.tol) || throw(ArgumentError("tol must be non-negative"))
    return nothing
end

"""
Reusable workspace for the ROF dual projection solver.
"""
struct ROFState{T<:AbstractFloat,N,A<:AbstractArray{T,N},G<:NTuple{N,A}}
    u::A
    u_prev::A
    divp::A
    g::A
    p::G
    grad_g::G
end

function ROFState(reference::AbstractArray{T,N}) where {T<:AbstractFloat,N}
    u = similar(reference)
    u_prev = similar(reference)
    divp = similar(reference)
    g = similar(reference)
    p = allocate_dual(reference)
    grad_g = allocate_dual(reference)
    fill!(u, zero(T))
    fill!(u_prev, zero(T))
    fill!(divp, zero(T))
    fill!(g, zero(T))
    for d = 1:N
        fill!(p[d], zero(T))
        fill!(grad_g[d], zero(T))
    end

    return ROFState{T,N,typeof(u),typeof(p)}(u, u_prev, divp, g, p, grad_g)
end

function _relative_change(
    u_prev::AbstractArray{T},
    u::AbstractArray{T},
) where {T<:AbstractFloat}
    return T(sqrt(sum(abs2, u .- u_prev)) / max(sqrt(sum(abs2, u_prev)), eps(T)))
end

function _rof_dual_step!(
    state::ROFState{T,N},
    problem::TVProblem{T,N},
    tau::T,
    inv_spacing::NTuple{N,T},
) where {T<:AbstractFloat,N}
    lambda = problem.lambda
    inv_lambda = inv(lambda)

    divergence!(state.divp, state.p, problem.boundary, inv_spacing)
    @. state.g = state.divp - inv_lambda * problem.f

    gradient!(state.grad_g, state.g, problem.boundary, inv_spacing)
    @inbounds for d = 1:N
        @. state.p[d] = state.p[d] + tau * state.grad_g[d]
    end

    project_dual_ball!(state.p, one(T), problem.tv_mode)

    divergence!(state.divp, state.p, problem.boundary, inv_spacing)
    @. state.u = problem.f - lambda * state.divp

    return nothing
end

function _tau_upper_bound(
    inv_spacing::NTuple{N,T},
    shape::NTuple{N,Int},
) where {N,T<:AbstractFloat}
    lipschitz_term = zero(T)
    @inbounds for d = 1:N
        if shape[d] > 1
            lipschitz_term += abs2(inv_spacing[d])
        end
    end

    lipschitz_term == zero(T) && return T(Inf)
    return inv(T(2) * lipschitz_term)
end

"""
Run ROF denoising in place.

`u` is both the initial guess and output buffer.
"""
function solve!(
    u::AbstractArray{T,N},
    problem::TVProblem{T,N},
    config::ROFConfig;
    state::Union{Nothing,ROFState{T,N}} = nothing,
) where {T<:AbstractFloat,N}
    _validate(config)
    problem.data_fidelity isa L2Fidelity ||
        throw(ArgumentError("ROF currently supports only L2Fidelity"))
    size(u) == size(problem.f) ||
        throw(ArgumentError("You must have the same size as problem.f"))

    if problem.lambda == zero(T)
        copyto!(u, problem.f)
        return SolverStats{T}(0, true, zero(T))
    end

    local_state = state === nothing ? ROFState(problem.f) : state

    copyto!(local_state.u, u)
    rel_change = T(Inf)
    converged = false
    iterations = config.maxiter
    tau_t = T(config.tau)
    inv_spacing = ntuple(d -> inv(problem.spacing[d]), Val(N))
    tau_upper = _tau_upper_bound(inv_spacing, size(problem.f))
    tau_t < tau_upper || throw(
        ArgumentError(
            "tau must be < $(tau_upper) for this grid spacing and shape; got $(tau_t)",
        ),
    )

    for k = 1:config.maxiter
        copyto!(local_state.u_prev, local_state.u)

        _rof_dual_step!(local_state, problem, tau_t, inv_spacing)

        if (k % config.check_every == 0) || (k == config.maxiter)
            rel_change = _relative_change(local_state.u_prev, local_state.u)
            if rel_change <= T(config.tol)
                converged = true
                iterations = k
                break
            end
        end

        iterations = k
    end

    copyto!(u, local_state.u)
    return SolverStats{T}(iterations, converged, rel_change)
end
