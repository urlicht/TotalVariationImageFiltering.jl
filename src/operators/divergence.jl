"""
In-place divergence operator that is negative-adjoint consistent with `gradient!`
under the same boundary convention.
"""
function divergence!(out::AbstractArray{T,N}, p::NTuple{N,AT}, ::Neumann) where {T,N,AT}
    fill!(out, zero(T))

    @inbounds for d in 1:N
        pd = p[d]
        n_d = size(out, d)
        n_d < 2 && continue # Skip dimensions with size 1

        # 1. The "Interior" and "Forward" contribution: 
        # For a backward difference (adjoint of forward diff), 
        # add p[i] to out[i] for all i from 1 to n-1.
        view_out_start = selectdim(out, d, 1:n_d-1)
        view_p_start = selectdim(pd, d, 1:n_d-1)
        view_out_start .+= view_p_start

        # 2. The "Backward" contribution: 
        # subtract p[i] from out[i+1] for all i from 1 to n-1.
        view_out_end = selectdim(out, d, 2:n_d)
        view_p_end = selectdim(pd, d, 1:n_d-1) # same p indices as above
        view_out_end .-= view_p_end
    end
    
    return out
end