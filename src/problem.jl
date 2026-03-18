"""
TV-based reconstruction/denoising
`min_u data_fidelity(u, f) + lambda * TV(u)`

`spacing[d]` is the physical grid spacing along axis `d` used by differential operators.
"""
struct TVProblem{
    T<:AbstractFloat,
    N,
    AF<:AbstractArray{T,N},
    DF<:AbstractDataFidelity,
    TV<:AbstractTVMode,
    BC<:AbstractBoundaryCondition,
}
    f::AF
    lambda::T
    spacing::NTuple{N,T}
    data_fidelity::DF
    tv_mode::TV
    boundary::BC
end

function _normalize_spacing(
    ::Type{T},
    ::Val{N},
    spacing::Nothing,
) where {T<:AbstractFloat,N}
    return ntuple(_ -> one(T), Val(N))
end

function _normalize_spacing(::Type{T}, ::Val{N}, spacing::Real) where {T<:AbstractFloat,N}
    scale = T(spacing)
    isfinite(scale) || throw(ArgumentError("spacing must be finite"))
    scale > zero(T) || throw(ArgumentError("spacing must be positive, got $spacing"))
    return ntuple(_ -> scale, Val(N))
end

function _normalize_spacing(
    ::Type{T},
    ::Val{N},
    spacing::NTuple{N,<:Real},
) where {T<:AbstractFloat,N}
    scales = ntuple(d -> T(spacing[d]), Val(N))
    @inbounds for d = 1:N
        isfinite(scales[d]) || throw(ArgumentError("spacing[$d] must be finite"))
        scales[d] > zero(T) ||
            throw(ArgumentError("spacing[$d] must be positive, got $(spacing[d])"))
    end
    return scales
end

function _normalize_spacing(
    ::Type{T},
    ::Val{N},
    spacing::AbstractVector{<:Real},
) where {T<:AbstractFloat,N}
    length(spacing) == N || throw(
        ArgumentError("spacing length must match ndims(f)=$N, got $(length(spacing))"),
    )
    scales = ntuple(d -> T(spacing[d]), Val(N))
    @inbounds for d = 1:N
        isfinite(scales[d]) || throw(ArgumentError("spacing[$d] must be finite"))
        scales[d] > zero(T) ||
            throw(ArgumentError("spacing[$d] must be positive, got $(spacing[d])"))
    end
    return scales
end

function _normalize_spacing(::Type{T}, ::Val{N}, spacing) where {T<:AbstractFloat,N}
    throw(
        ArgumentError(
            "spacing must be `nothing`, a real scalar, an NTuple{$N,<:Real}, or an AbstractVector{<:Real}",
        ),
    )
end

function TVProblem(
    f::AbstractArray{T,N};
    lambda::Real,
    spacing = nothing,
    data_fidelity::AbstractDataFidelity = L2Fidelity(),
    tv_mode::AbstractTVMode = IsotropicTV(),
    boundary::AbstractBoundaryCondition = Neumann(),
) where {T<:AbstractFloat,N}
    lambda_t = T(lambda)
    lambda_t >= zero(T) || throw(ArgumentError("lambda must be non-negative"))
    spacing_t = _normalize_spacing(T, Val(N), spacing)
    return TVProblem{T,N,typeof(f),typeof(data_fidelity),typeof(tv_mode),typeof(boundary)}(
        f,
        lambda_t,
        spacing_t,
        data_fidelity,
        tv_mode,
        boundary,
    )
end
