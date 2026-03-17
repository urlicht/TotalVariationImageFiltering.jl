"""
In-place forward finite-difference gradient with Neumann boundary treatment.

For each dimension `d`,
`g[d][i] = u[i + e_d] - u[i]` on interior points and `0` on the upper boundary.
"""
function gradient!(g::NTuple{N,AT}, u::AbstractArray{T,N}, ::Neumann) where {T,N,AT<:AbstractArray{T,N}}
    # @views ensures all slicing uses view instead of copy
    @views for d in 1:N
        gd = g[d]
        n_d = size(u, d)
                
        # 1. interior differences
        target = selectdim(gd, d, 1:n_d-1)
        u_front = selectdim(u, d, 1:n_d-1)
        u_back = selectdim(u, d, 2:n_d)
        
        target .= u_back .- u_front
        
        # 2. Neumann boundary to zero
        selectdim(gd, d, n_d) .= zero(T)
    end
    
    return g
end