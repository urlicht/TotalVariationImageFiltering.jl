"""
TV-based reconstruction/denoising
`min_u data_fidelity(u, f) + lambda * TV(u)`
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
    data_fidelity::DF
    tv_mode::TV
    boundary::BC
end

function TVProblem(
    f::AbstractArray{T,N};
    lambda::Real,
    data_fidelity::AbstractDataFidelity = L2Fidelity(),
    tv_mode::AbstractTVMode = IsotropicTV(),
    boundary::AbstractBoundaryCondition = Neumann(),
) where {T<:AbstractFloat,N}
    lambda_t = T(lambda)
    lambda_t >= zero(T) || throw(ArgumentError("lambda must be non-negative"))
    return TVProblem{T,N,typeof(f),typeof(data_fidelity),typeof(tv_mode),typeof(boundary)}(
        f,
        lambda_t,
        data_fidelity,
        tv_mode,
        boundary,
    )
end
