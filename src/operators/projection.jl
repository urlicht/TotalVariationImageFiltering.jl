"""
Project dual field `p` onto the isotropic TV dual ball:

`||p[i]||_2 <= radius` for every pixel/voxel `i`.
"""
function project_dual_ball!(p::NTuple{N,AT}, radius::T, ::IsotropicTV) where {N,T,AT}
    radius2 = radius * radius

    # We use a single loop over the underlying data to maximize SIMD
    # This assumes all arrays in the NTuple have the same layout
    @inbounds for i in eachindex(p[1])
        # 1. Manually unroll the norm calculation for speed
        nrm2 = zero(T)
        for d = 1:N
            val = p[d][i]
            nrm2 = muladd(val, val, nrm2)
        end

        # 2. Only perform the math/writes if necessary
        if nrm2 > radius2
            nrm = sqrt(nrm2)
            scale = radius / nrm
            for d = 1:N
                p[d][i] *= scale
            end
        end
    end

    return p
end

"""
Project dual field `p` onto the anisotropic TV dual ball:

`|p[d][i]| <= radius` for every component and pixel/voxel `i`.
"""
function project_dual_ball!(p::NTuple{N,AT}, radius::T, ::AnisotropicTV) where {N,T,AT}
    lower = -radius
    upper = radius

    @inbounds for d = 1:N
        pd = p[d]
        for i in eachindex(pd)
            pd[i] = clamp(pd[i], lower, upper)
        end
    end

    return p
end
