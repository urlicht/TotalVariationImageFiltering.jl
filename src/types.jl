"""
Abstract supertype for boundary conventions used by differential operators.
"""
abstract type AbstractBoundaryCondition end

"""
Forward-difference gradient with homogeneous Neumann boundary treatment.
"""
struct Neumann <: AbstractBoundaryCondition end

"""
Abstract supertype for data-fidelity terms.
"""
abstract type AbstractDataFidelity end

"""
L2 data fidelity, `0.5 * ||u - f||^2`.
"""
struct L2Fidelity <: AbstractDataFidelity end

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

