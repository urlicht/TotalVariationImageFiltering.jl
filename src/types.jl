#
# Boundary condition
#

"""
Abstract supertype for boundary conventions used by differential operators.
"""
abstract type AbstractBoundaryCondition end

"""
Forward-difference gradient with homogeneous Neumann boundary treatment.
"""
struct Neumann <: AbstractBoundaryCondition end

#
# Data fidelity
#

"""
Abstract supertype for data-fidelity terms.
"""
abstract type AbstractDataFidelity end

"""
L2 data fidelity, `0.5 * ||u - f||^2`.
"""
struct L2Fidelity <: AbstractDataFidelity end

#
# TVMode (pixel/voxel) isotropy
#

"""
Abstract supertype for TV penalty modes.
"""
abstract type AbstractTVMode end

"""
Isotropic TV, `sum(sqrt(sum_d (grad_d u)^2))`.
"""
struct IsotropicTV <: AbstractTVMode end

"""
Anisotropic TV, `sum(sum_d |grad_d u|)`.
"""
struct AnisotropicTV <: AbstractTVMode end

#
# Solver
#
"""
Abstract supertype for all total-variation solver configurations.
"""
abstract type AbstractTVSolver end

"""
Convergence summary for one solver run.
"""
struct SolverStats{T<:AbstractFloat}
    iterations::Int
    converged::Bool
    rel_change::T
end

function _validate_common_config(maxiter::Int, check_every::Int)
    maxiter > 0 || throw(ArgumentError("maxiter must be positive, got $maxiter"))
    check_every > 0 ||
        throw(ArgumentError("check_every must be positive, got $check_every"))
    return nothing
end
