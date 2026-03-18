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

function _dual_update_if_active_kernel!(
    pd,
    grad_pd,
    tau,
    done_mask,
    batch_stride::Int,
    batch_size::Int,
    len::Int,
)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if index <= len
        i = Int(index)
        batch_idx = ((i - 1) ÷ batch_stride) % batch_size + 1
        @inbounds if done_mask[batch_idx] == UInt8(0)
            pd[i] = pd[i] + tau * grad_pd[i]
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
    done_mask::CUDA.CuArray{UInt8,1},
    has_done::Bool,
    batch_stride::Int,
    batch_size::Int,
) where {T<:AbstractFloat,N,M}
    inv_lambda = inv(lambda)

    _divergence_spatial!(state.divp, state.p, inv_spacing)
    @. state.g = state.divp - inv_lambda * f_batch

    _gradient_spatial!(state.grad_g, state.g, inv_spacing)
    if has_done
        len = length(state.p[1])
        len > 0 && batch_size > 0 && batch_stride > 0 || return nothing
        threads, blocks = _launch_config(len)
        @inbounds for d = 1:M
            CUDA.@cuda threads = threads blocks = blocks _dual_update_if_active_kernel!(
                state.p[d],
                state.grad_g[d],
                tau,
                done_mask,
                batch_stride,
                batch_size,
                len,
            )
        end
    else
        @inbounds for d = 1:M
            @. state.p[d] = state.p[d] + tau * state.grad_g[d]
        end
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
    batch_count = size(f_batch, N)
    if batch_count == 0
        copyto!(u_batch, f_batch)
        return SolverStats{T}(0, true, zero(T))
    end

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
    done_host = zeros(UInt8, batch_count)
    done_device = CUDA.zeros(UInt8, batch_count)
    rel_changes = fill(T(Inf), batch_count)
    iterations_per_slice = fill(config.maxiter, batch_count)
    active_count = batch_count
    has_done = false
    batch_stride = stride(f_batch, N)

    for k = 1:config.maxiter
        copyto!(local_state.u_prev, local_state.u)
        _rof_batch_dual_step!(
            local_state,
            f_batch,
            tau_t,
            lambda_t,
            inv_spacing,
            tv_mode,
            done_device,
            has_done,
            batch_stride,
            batch_count,
        )

        if (k % config.check_every == 0) || (k == config.maxiter)
            mask_changed = false
            @views for b = 1:batch_count
                done_host[b] == UInt8(1) && continue
                rel = TVImageFiltering._relative_change(
                    selectdim(local_state.u_prev, N, b),
                    selectdim(local_state.u, N, b),
                )
                rel_changes[b] = rel
                if rel <= T(config.tol)
                    done_host[b] = UInt8(1)
                    iterations_per_slice[b] = k
                    active_count -= 1
                    mask_changed = true
                end
            end

            if active_count == 0
                copyto!(u_batch, local_state.u)
                return SolverStats{T}(
                    maximum(iterations_per_slice),
                    true,
                    maximum(rel_changes),
                )
            end

            if mask_changed
                has_done = true
                copyto!(done_device, done_host)
            end
        end
    end

    copyto!(u_batch, local_state.u)
    return SolverStats{T}(maximum(iterations_per_slice), false, maximum(rel_changes))
end

end # module
