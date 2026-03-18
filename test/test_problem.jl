using Test
using Random
using TVImageFiltering

struct DummyBoundaryForProblem <: TVImageFiltering.AbstractBoundaryCondition end
struct DummyDataFidelityForProblem <: TVImageFiltering.AbstractDataFidelity end
struct DummyTVModeForProblem <: TVImageFiltering.AbstractTVMode end
struct DummyConstraintForProblem <: TVImageFiltering.AbstractPrimalConstraint end

@testset "TVProblem Construction and Defaults" begin
    Random.seed!(11)
    f = randn(Float32, 3, 4)
    prob = TVImageFiltering.TVProblem(f; lambda = 2)

    @test prob.f === f
    @test prob.lambda isa Float32
    @test prob.lambda == 2.0f0
    @test prob.spacing == (1.0f0, 1.0f0)
    @test prob.data_fidelity isa TVImageFiltering.L2Fidelity
    @test prob.tv_mode isa TVImageFiltering.IsotropicTV
    @test prob.boundary isa TVImageFiltering.Neumann
    @test prob.constraint isa TVImageFiltering.NoConstraint
end

@testset "TVProblem Custom Components" begin
    f = randn(Float64, 4, 2)
    df = DummyDataFidelityForProblem()
    tv = DummyTVModeForProblem()
    bc = DummyBoundaryForProblem()
    constraint = TVImageFiltering.BoxConstraint(-2.0, 1.0)

    prob = TVImageFiltering.TVProblem(
        f;
        lambda = 0.3,
        spacing = (0.5, 3.0),
        data_fidelity = df,
        tv_mode = tv,
        boundary = bc,
        constraint = constraint,
    )

    @test prob.data_fidelity === df
    @test prob.tv_mode === tv
    @test prob.boundary === bc
    @test prob.constraint isa TVImageFiltering.BoxConstraint{Float64}
    @test prob.constraint.lower == -2.0
    @test prob.constraint.upper == 1.0
    @test prob.spacing == (0.5, 3.0)
end

@testset "TVProblem Constraint Normalization and Validation" begin
    f32 = randn(Float32, 5, 4)
    prob_box32 = TVImageFiltering.TVProblem(
        f32;
        lambda = 0.2f0,
        constraint = TVImageFiltering.BoxConstraint(-1.0, 2.0),
    )
    @test prob_box32.constraint isa TVImageFiltering.BoxConstraint{Float32}
    @test prob_box32.constraint.lower == -1.0f0
    @test prob_box32.constraint.upper == 2.0f0

    prob_nn = TVImageFiltering.TVProblem(
        f32;
        lambda = 0.2f0,
        constraint = TVImageFiltering.NonnegativeConstraint(),
    )
    @test prob_nn.constraint isa TVImageFiltering.NonnegativeConstraint

    @test_throws ArgumentError TVImageFiltering.TVProblem(
        f32;
        lambda = 0.2f0,
        constraint = DummyConstraintForProblem(),
    )
end

@testset "TVProblem Spacing Normalization" begin
    f = randn(Float64, 3, 4, 2)

    prob_default = TVImageFiltering.TVProblem(f; lambda = 0.1)
    @test prob_default.spacing == (1.0, 1.0, 1.0)

    prob_scalar = TVImageFiltering.TVProblem(f; lambda = 0.1, spacing = 2)
    @test prob_scalar.spacing == (2.0, 2.0, 2.0)

    prob_tuple = TVImageFiltering.TVProblem(f; lambda = 0.1, spacing = (0.5, 2.0, 4.0))
    @test prob_tuple.spacing == (0.5, 2.0, 4.0)

    prob_vector = TVImageFiltering.TVProblem(f; lambda = 0.1, spacing = [0.5, 2.0, 4.0])
    @test prob_vector.spacing == (0.5, 2.0, 4.0)

    @test_throws ArgumentError TVImageFiltering.TVProblem(
        f;
        lambda = 0.1,
        spacing = [1.0, 2.0],
    )
    @test_throws ArgumentError TVImageFiltering.TVProblem(
        f;
        lambda = 0.1,
        spacing = (1.0, 0.0, 2.0),
    )
    @test_throws ArgumentError TVImageFiltering.TVProblem(
        f;
        lambda = 0.1,
        spacing = (1.0, -1.0, 2.0),
    )
    @test_throws ArgumentError TVImageFiltering.TVProblem(
        f;
        lambda = 0.1,
        spacing = (1.0, Inf, 2.0),
    )
    @test_throws ArgumentError TVImageFiltering.TVProblem(
        f;
        lambda = 0.1,
        spacing = "invalid",
    )
end

@testset "TVProblem Lambda Validation" begin
    f = randn(Float64, 3, 4)
    @test_throws ArgumentError TVImageFiltering.TVProblem(f; lambda = -1.0)
end
