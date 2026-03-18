"""
Configuration for the ROF dual-projection solver.

This solver targets the Rudin-Osher-Fatemi denoising model with `L2Fidelity`:

    minimize_u 0.5 * ||u - f||_2^2 + lambda * TV(u)

`tau` is the dual ascent step and must satisfy the stability bound used by this
implementation:

    tau < 1 / (2 * sum_{d: n_d > 1} h_d^(-2))

where `h_d = spacing[d]` and `n_d = size(f, d)`.

Fields:
- `maxiter`: maximum number of iterations.
- `tau`: dual step size.
- `tol`: stopping tolerance on relative primal change.
- `check_every`: evaluate convergence every `check_every` iterations.

References:
- L. I. Rudin, S. Osher, E. Fatemi, "Nonlinear total variation based noise
  removal algorithms," *Physica D* 60(1-4):259-268, 1992.
  DOI: 10.1016/0167-2789(92)90242-F
- A. Chambolle, "An algorithm for total variation minimization and
  applications," *Journal of Mathematical Imaging and Vision* 20:89-97, 2004.
  DOI: 10.1023/B:JMIV.0000011325.36760.1E
"""
struct ROFConfig{T<:AbstractFloat} <: AbstractTVSolver
    maxiter::Int
    tau::T
    tol::T
    check_every::Int
end

function ROFConfig(;
    maxiter::Int = 300,
    tau::Real = 0.0625, # Conservative default; for 2D unit spacing, Chambolle (2004) gives tau < 0.25.
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

The implementation follows Chambolle's dual projection method for the ROF model.
With dual variable `p` and image `f`, each iteration computes:

    g^k = div(p^k) - f / lambda

    p^(k+1) = Proj_B(p^k + tau * grad(g^k))

    u^(k+1) = f - lambda * div(p^(k+1))

where `Proj_B` is projection onto the TV dual unit ball:
- isotropic TV: `||p[i]||_2 <= 1` per pixel/voxel,
- anisotropic TV: `|p[d][i]| <= 1` per component.

`gradient!` and `divergence!` use forward/backward finite differences with
homogeneous Neumann boundary handling and account for `problem.spacing`.

References:
- L. I. Rudin, S. Osher, E. Fatemi, "Nonlinear total variation based noise
  removal algorithms," *Physica D* 60(1-4):259-268, 1992.
  DOI: 10.1016/0167-2789(92)90242-F
- A. Chambolle, "An algorithm for total variation minimization and
  applications," *Journal of Mathematical Imaging and Vision* 20:89-97, 2004.
  DOI: 10.1023/B:JMIV.0000011325.36760.1E
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
