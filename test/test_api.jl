using Test
using Random
using TVImageFiltering

struct DummySolverForAPI <: TVImageFiltering.AbstractTVSolver end

@testset "solve API Default Path and Init Semantics" begin
    Random.seed!(101)
    f = randn(10)
    prob = TVImageFiltering.TVProblem(f; lambda = 0.0)
    init = fill(9.0, size(f))
    init_before = copy(init)

    u, stats = TVImageFiltering.solve(prob, TVImageFiltering.ROFConfig(); init = init)
    @test u == f
    @test stats.converged
    @test u !== init
    @test u !== f
    @test init == init_before

    u_default, stats_default = TVImageFiltering.solve(prob)
    @test u_default == f
    @test stats_default isa TVImageFiltering.SolverStats{Float64}
end

@testset "solve API Init Validation" begin
    f = randn(5, 4)
    prob = TVImageFiltering.TVProblem(f; lambda = 0.1)
    @test_throws ArgumentError TVImageFiltering.solve(
        prob,
        TVImageFiltering.ROFConfig();
        init = zeros(5, 5),
    )
end

@testset "solve API Keyword Forwarding to solve!" begin
    Random.seed!(103)
    f = randn(20)
    prob = TVImageFiltering.TVProblem(
        f;
        lambda = 0.2,
        tv_mode = TVImageFiltering.AnisotropicTV(),
    )
    cfg = TVImageFiltering.ROFConfig(
        maxiter = 2000,
        tau = 0.0625,
        tol = 1e-8,
        check_every = 20,
    )
    state = TVImageFiltering.ROFState(f)

    u1, s1 = TVImageFiltering.solve(prob, cfg; state = state)
    u2, s2 = TVImageFiltering.solve(prob, cfg; init = fill(3.0, size(f)), state = state)
    @test s1.converged
    @test s2.converged
    @test isapprox(u1, u2; atol = 1e-6, rtol = 0.0)
end

@testset "solve!/solve Fallback for Unknown Solver" begin
    f = randn(6)
    prob = TVImageFiltering.TVProblem(f; lambda = 0.1)
    u = zeros(6)

    @test_throws MethodError TVImageFiltering.solve!(u, prob, DummySolverForAPI())
    @test_throws MethodError TVImageFiltering.solve(prob, DummySolverForAPI())
end
