module TVImageFilteringCUDAExt

using CUDA
using TVImageFiltering

import TVImageFiltering:
    AnisotropicTV, IsotropicTV, Neumann, divergence!, gradient!, project_dual_ball!

const DEFAULT_THREADS = 256

@inline function _launch_config(len::Int)
    return DEFAULT_THREADS, cld(len, DEFAULT_THREADS)
end

function _gradient_dim_kernel!(gd, u, scale, stride_d::Int, n_d::Int, len::Int)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if index <= len
        i = Int(index)
        coord_d = ((i - 1) ÷ stride_d) % n_d + 1
        @inbounds if coord_d < n_d
            gd[i] = scale * (u[i+stride_d] - u[i])
        else
            gd[i] = zero(scale)
        end
    end
    return nothing
end

function _divergence_dim_kernel!(out, pd, scale, stride_d::Int, n_d::Int, len::Int)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if index <= len
        i = Int(index)
        coord_d = ((i - 1) ÷ stride_d) % n_d + 1
        acc = zero(scale)
        @inbounds if coord_d < n_d
            acc += scale * pd[i]
        end
        @inbounds if coord_d > 1
            acc -= scale * pd[i-stride_d]
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

function gradient!(
    g::NTuple{N,AT},
    u::CUDA.CuArray{T,N},
    ::Neumann,
    inv_spacing::NTuple{N,S},
) where {T<:AbstractFloat,N,S<:Real,AT<:CUDA.CuArray{T,N}}
    len = length(u)
    len == 0 && return g

    threads, blocks = _launch_config(len)
    strides = ntuple(d -> stride(u, d), Val(N))

    @inbounds for d = 1:N
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

function divergence!(
    out::CUDA.CuArray{T,N},
    p::NTuple{N,AT},
    ::Neumann,
    inv_spacing::NTuple{N,S},
) where {T<:AbstractFloat,N,S<:Real,AT<:CUDA.CuArray{T,N}}
    fill!(out, zero(T))
    len = length(out)
    len == 0 && return out

    threads, blocks = _launch_config(len)
    strides = ntuple(d -> stride(out, d), Val(N))

    @inbounds for d = 1:N
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

end # module
