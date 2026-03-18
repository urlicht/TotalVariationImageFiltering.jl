"""
Allocate a tuple of `N` arrays with the same shape/type as `u` to store vector-valued fields.
"""
function allocate_dual(u::AbstractArray{T,N}) where {T,N}
    return ntuple(_ -> similar(u), Val(N))
end
