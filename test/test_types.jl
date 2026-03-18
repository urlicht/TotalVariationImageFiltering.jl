using Test
using TVImageFiltering

@testset "Type Hierarchy" begin
    @test TVImageFiltering.Neumann <: TVImageFiltering.AbstractBoundaryCondition
    @test TVImageFiltering.L2Fidelity <: TVImageFiltering.AbstractDataFidelity
    @test TVImageFiltering.PoissonFidelity <: TVImageFiltering.AbstractDataFidelity
    @test TVImageFiltering.NoConstraint <: TVImageFiltering.AbstractPrimalConstraint
    @test TVImageFiltering.NonnegativeConstraint <: TVImageFiltering.AbstractPrimalConstraint
    @test TVImageFiltering.BoxConstraint <: TVImageFiltering.AbstractPrimalConstraint
    @test TVImageFiltering.IsotropicTV <: TVImageFiltering.AbstractTVMode
    @test TVImageFiltering.AnisotropicTV <: TVImageFiltering.AbstractTVMode
    @test TVImageFiltering.ROFConfig <: TVImageFiltering.AbstractTVSolver
    @test TVImageFiltering.PDHGConfig <: TVImageFiltering.AbstractTVSolver
end

@testset "Constraint Constructors" begin
    no_constraint = TVImageFiltering.NoConstraint()
    @test no_constraint isa TVImageFiltering.NoConstraint

    nn_constraint = TVImageFiltering.NonnegativeConstraint()
    @test nn_constraint isa TVImageFiltering.NonnegativeConstraint

    box = TVImageFiltering.BoxConstraint(-1.0, 2.5f0)
    @test box isa TVImageFiltering.BoxConstraint{Float64}
    @test box.lower == -1.0
    @test box.upper == 2.5

    @test_throws ArgumentError TVImageFiltering.BoxConstraint(1.0, -1.0)
    @test_throws ArgumentError TVImageFiltering.BoxConstraint(NaN, 1.0)
    @test_throws ArgumentError TVImageFiltering.BoxConstraint(-1.0, NaN)
end

@testset "Common Config Validation" begin
    @test TVImageFiltering._validate_common_config(1, 1) === nothing
    @test_throws ArgumentError TVImageFiltering._validate_common_config(0, 1)
    @test_throws ArgumentError TVImageFiltering._validate_common_config(1, 0)
end

@testset "PDHGConfig Constructor and Validation" begin
    cfg_default = TVImageFiltering.PDHGConfig()
    @test cfg_default.maxiter == 500
    @test cfg_default.tau == 0.25
    @test cfg_default.sigma == 0.25
    @test cfg_default.theta == 1.0
    @test cfg_default.tol == 1e-4
    @test cfg_default.check_every == 10
    @test TVImageFiltering._validate(cfg_default) === nothing

    cfg_promoted = TVImageFiltering.PDHGConfig(tau = Float32(0.2), sigma = 0.1, tol = 1e-6)
    @test cfg_promoted isa TVImageFiltering.PDHGConfig{Float64}

    cfg_float32 = TVImageFiltering.PDHGConfig(
        tau = Float32(0.2),
        sigma = Float32(0.1),
        theta = Float32(1),
        tol = Float32(1e-3),
    )
    @test cfg_float32 isa TVImageFiltering.PDHGConfig{Float32}

    @test_throws ArgumentError TVImageFiltering._validate(
        TVImageFiltering.PDHGConfig(maxiter = 0),
    )
    @test_throws ArgumentError TVImageFiltering._validate(
        TVImageFiltering.PDHGConfig(check_every = 0),
    )
    @test_throws ArgumentError TVImageFiltering._validate(
        TVImageFiltering.PDHGConfig(tau = 0.0),
    )
    @test_throws ArgumentError TVImageFiltering._validate(
        TVImageFiltering.PDHGConfig(sigma = 0.0),
    )
    @test_throws ArgumentError TVImageFiltering._validate(
        TVImageFiltering.PDHGConfig(theta = -0.1),
    )
    @test_throws ArgumentError TVImageFiltering._validate(
        TVImageFiltering.PDHGConfig(theta = 1.1),
    )
    @test_throws ArgumentError TVImageFiltering._validate(
        TVImageFiltering.PDHGConfig(tol = -1e-6),
    )
end

@testset "SolverStats Shape" begin
    stats = TVImageFiltering.SolverStats{Float64}(17, true, 1e-6)
    @test stats.iterations == 17
    @test stats.converged
    @test stats.rel_change == 1e-6
end

@testset "ROFConfig Constructor and Validation" begin
    cfg_default = TVImageFiltering.ROFConfig()
    @test cfg_default.maxiter == 300
    @test cfg_default.tau == 0.0625
    @test cfg_default.tol == 1e-4
    @test cfg_default.check_every == 10
    @test TVImageFiltering._validate(cfg_default) === nothing

    cfg_promoted = TVImageFiltering.ROFConfig(tau = Float32(0.2), tol = 1e-6)
    @test cfg_promoted isa TVImageFiltering.ROFConfig{Float64}

    cfg_float32 = TVImageFiltering.ROFConfig(tau = Float32(0.2), tol = Float32(1e-3))
    @test cfg_float32 isa TVImageFiltering.ROFConfig{Float32}

    @test_throws ArgumentError TVImageFiltering._validate(
        TVImageFiltering.ROFConfig(maxiter = 0),
    )
    @test_throws ArgumentError TVImageFiltering._validate(
        TVImageFiltering.ROFConfig(check_every = 0),
    )
    @test_throws ArgumentError TVImageFiltering._validate(
        TVImageFiltering.ROFConfig(tau = 0.0),
    )
    @test_throws ArgumentError TVImageFiltering._validate(
        TVImageFiltering.ROFConfig(tol = -1e-6),
    )
end
