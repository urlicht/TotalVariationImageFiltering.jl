module TVImageFilteringCUDAExt

using CUDA
using TVImageFiltering

import TVImageFiltering:
    AbstractBoundaryCondition,
    AbstractDataFidelity,
    AbstractTVMode,
    AnisotropicTV,
    IsotropicTV,
    L2Fidelity,
    Neumann,
    ROFConfig,
    SolverStats,
    divergence!,
    gradient!,
    project_dual_ball!,
    solve_batch!

const DEFAULT_THREADS = 256

@inline function _launch_config(len::Int)
    return DEFAULT_THREADS, cld(len, DEFAULT_THREADS)
end

function _gradient_dim_kernel!(
    gd,
    u,
    scale,
    stride_d::Int,
    n_d::Int,
    len::Int,
)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if index <= len
        i = Int(index)
        coord_d = ((i - 1) ÷ stride_d) % n_d + 1
        @inbounds if coord_d < n_d
            gd[i] = scale * (u[i + stride_d] - u[i])
        else
            gd[i] = zero(scale)
        end
    end
    return nothing
end

function _divergence_dim_kernel!(
    out,
    pd,
    scale,
    stride_d::Int,
    n_d::Int,
    len::Int,
)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if index <= len
        i = Int(index)
        coord_d = ((i - 1) ÷ stride_d) % n_d + 1
        acc = zero(scale)
        @inbounds if coord_d < n_d
            acc += scale * pd[i]
        end
        @inbounds if coord_d > 1
            acc -= scale * pd[i - stride_d]
        end
        @inbounds out[i] += acc
    end
    return nothing
end

function _project_isotropic_kernel!(p, radius, radius2, len::Int)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if index <= len
        i = Int(index)
        nrm2 = zero(radius)
        @inbounds for d = 1:length(p)
            v = p[d][i]
            nrm2 = muladd(v, v, nrm2)
        end

        @inbounds if nrm2 > radius2
            inv_nrm = radius / sqrt(nrm2)
            for d = 1:length(p)
                p[d][i] *= inv_nrm
            end
        end
    end
    return nothing
end

function _project_anisotropic_kernel!(p, lower, upper, len::Int)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if index <= len
        i = Int(index)
        @inbounds for d = 1:length(p)
            p[d][i] = clamp(p[d][i], lower, upper)
        end
    end
    return nothing
end

function _gradient_spatial!(
    g::NTuple{M,AT},
    u::CUDA.CuArray{T,N},
    inv_spacing::NTuple{M,S},
) where {T<:AbstractFloat,N,M,S<:Real,AT<:CUDA.CuArray{T,N}}
    len = length(u)
    len == 0 && return g

    threads, blocks = _launch_config(len)
    strides = ntuple(d -> stride(u, d), Val(M))

    @inbounds for d = 1:M
        CUDA.@cuda threads = threads blocks = blocks _gradient_dim_kernel!(
            g[d],
            u,
            T(inv_spacing[d]),
            strides[d],
            size(u, d),
            len,
        )
    end

    return g
end

function _divergence_spatial!(
    out::CUDA.CuArray{T,N},
    p::NTuple{M,AT},
    inv_spacing::NTuple{M,S},
) where {T<:AbstractFloat,N,M,S<:Real,AT<:CUDA.CuArray{T,N}}
    fill!(out, zero(T))
    len = length(out)
    len == 0 && return out

    threads, blocks = _launch_config(len)
    strides = ntuple(d -> stride(out, d), Val(M))

    @inbounds for d = 1:M
        CUDA.@cuda threads = threads blocks = blocks _divergence_dim_kernel!(
            out,
            p[d],
            T(inv_spacing[d]),
            strides[d],
            size(out, d),
            len,
        )
    end

    return out
end

function gradient!(
    g::NTuple{N,AT},
    u::CUDA.CuArray{T,N},
    ::Neumann,
    inv_spacing::NTuple{N,S},
) where {T<:AbstractFloat,N,S<:Real,AT<:CUDA.CuArray{T,N}}
    return _gradient_spatial!(g, u, inv_spacing)
end

function divergence!(
    out::CUDA.CuArray{T,N},
    p::NTuple{N,AT},
    ::Neumann,
    inv_spacing::NTuple{N,S},
) where {T<:AbstractFloat,N,S<:Real,AT<:CUDA.CuArray{T,N}}
    return _divergence_spatial!(out, p, inv_spacing)
end

function project_dual_ball!(
    p::NTuple{N,AT},
    radius::T,
    ::IsotropicTV,
) where {N,T<:AbstractFloat,AT<:CUDA.CuArray{T}}
    len = length(p[1])
    len == 0 && return p
    threads, blocks = _launch_config(len)
    radius2 = radius * radius
    CUDA.@cuda threads = threads blocks = blocks _project_isotropic_kernel!(
        p,
        radius,
        radius2,
        len,
    )
    return p
end

function project_dual_ball!(
    p::NTuple{N,AT},
    radius::T,
    ::AnisotropicTV,
) where {N,T<:AbstractFloat,AT<:CUDA.CuArray{T}}
    len = length(p[1])
    len == 0 && return p
    threads, blocks = _launch_config(len)
    lower = -radius
    upper = radius
    CUDA.@cuda threads = threads blocks = blocks _project_anisotropic_kernel!(
        p,
        lower,
        upper,
        len,
    )
    return p
end

struct ROFBatchState{T<:AbstractFloat,N,M,A<:CUDA.CuArray{T,N},G<:NTuple{M,A}}
    u::A
    u_prev::A
    divp::A
    g::A
    p::G
    grad_g::G
end

function ROFBatchState(reference::CUDA.CuArray{T,N}, ::Val{M}) where {T<:AbstractFloat,N,M}
    u = similar(reference)
    u_prev = similar(reference)
    divp = similar(reference)
    g = similar(reference)
    p = ntuple(_ -> similar(reference), Val(M))
    grad_g = ntuple(_ -> similar(reference), Val(M))

    fill!(u, zero(T))
    fill!(u_prev, zero(T))
    fill!(divp, zero(T))
    fill!(g, zero(T))
    @inbounds for d = 1:M
        fill!(p[d], zero(T))
        fill!(grad_g[d], zero(T))
    end

    return ROFBatchState{T,N,M,typeof(u),typeof(p)}(u, u_prev, divp, g, p, grad_g)
end

function _rof_batch_dual_step!(
    state::ROFBatchState{T,N,M},
    f_batch::CUDA.CuArray{T,N},
    tau::T,
    lambda::T,
    inv_spacing::NTuple{M,T},
    tv_mode::AbstractTVMode,
) where {T<:AbstractFloat,N,M}
    inv_lambda = inv(lambda)

    _divergence_spatial!(state.divp, state.p, inv_spacing)
    @. state.g = state.divp - inv_lambda * f_batch

    _gradient_spatial!(state.grad_g, state.g, inv_spacing)
    @inbounds for d = 1:M
        @. state.p[d] = state.p[d] + tau * state.grad_g[d]
    end

    project_dual_ball!(state.p, one(T), tv_mode)

    _divergence_spatial!(state.divp, state.p, inv_spacing)
    @. state.u = f_batch - lambda * state.divp
    return nothing
end

function solve_batch!(
    u_batch::CUDA.CuArray{T,N},
    f_batch::CUDA.CuArray{T,N},
    config::ROFConfig;
    lambda::Real,
    spacing = nothing,
    data_fidelity::AbstractDataFidelity = L2Fidelity(),
    tv_mode::AbstractTVMode = IsotropicTV(),
    boundary::AbstractBoundaryCondition = Neumann(),
    state = nothing,
) where {T<:AbstractFloat,N}
    N >= 2 ||
        throw(ArgumentError("f_batch must have at least 2 dimensions (spatial..., batch)"))
    size(u_batch) == size(f_batch) ||
        throw(ArgumentError("u_batch and f_batch must have matching sizes"))

    TVImageFiltering._validate(config)
    data_fidelity isa L2Fidelity ||
        throw(ArgumentError("ROF currently supports only L2Fidelity"))
    boundary isa Neumann ||
        throw(ArgumentError("ROF batch CUDA currently supports only Neumann boundary"))

    lambda_t = T(lambda)
    lambda_t >= zero(T) || throw(ArgumentError("lambda must be non-negative"))
    if lambda_t == zero(T)
        copyto!(u_batch, f_batch)
        return SolverStats{T}(0, true, zero(T))
    end

    spatial_ndims = N - 1
    spacing_t = TVImageFiltering._normalize_spacing(T, Val(spatial_ndims), spacing)
    inv_spacing = ntuple(d -> inv(spacing_t[d]), Val(spatial_ndims))
    shape_spatial = ntuple(d -> size(f_batch, d), Val(spatial_ndims))

    tau_t = T(config.tau)
    tau_upper = TVImageFiltering._tau_upper_bound(inv_spacing, shape_spatial)
    tau_t < tau_upper || throw(
        ArgumentError(
            "tau must be < $(tau_upper) for this grid spacing and shape; got $(tau_t)",
        ),
    )

    local_state = if state === nothing
        ROFBatchState(f_batch, Val(spatial_ndims))
    elseif state isa ROFBatchState
        size(state.u) == size(f_batch) ||
            throw(ArgumentError("state buffers must match f_batch size"))
        length(state.p) == spatial_ndims ||
            throw(ArgumentError("state dual buffers must match spatial dimensions"))
        state
    else
        throw(
            ArgumentError(
                "state must be `nothing` or a compatible ROFBatchState object",
            ),
        )
    end

    copyto!(local_state.u, u_batch)
    rel_change = T(Inf)
    converged = false
    iterations = config.maxiter

    for k = 1:config.maxiter
        copyto!(local_state.u_prev, local_state.u)
        _rof_batch_dual_step!(
            local_state,
            f_batch,
            tau_t,
            lambda_t,
            inv_spacing,
            tv_mode,
        )

        if (k % config.check_every == 0) || (k == config.maxiter)
            rel_change = TVImageFiltering._relative_change(local_state.u_prev, local_state.u)
            if rel_change <= T(config.tol)
                converged = true
                iterations = k
                break
            end
        end

        iterations = k
    end

    copyto!(u_batch, local_state.u)
    return SolverStats{T}(iterations, converged, rel_change)
end

end # module
