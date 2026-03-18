using Test
using Random
using TVImageFiltering

@testset "solve_batch CPU API" begin
    Random.seed!(503)
    f_batch = randn(Float64, 16, 12, 4)
    config = TVImageFiltering.ROFConfig(maxiter = 1500, tau = 0.0625, tol = 1e-8, check_every = 10)

    u_batch, stats_batch =
        TVImageFiltering.solve_batch(f_batch, config; lambda = 0.2, tv_mode = TVImageFiltering.IsotropicTV())

    expected = similar(f_batch)
    per_stats = TVImageFiltering.SolverStats{Float64}[]
    @views for b = 1:size(f_batch, 3)
        fb = selectdim(f_batch, 3, b)
        prob = TVImageFiltering.TVProblem(fb; lambda = 0.2, tv_mode = TVImageFiltering.IsotropicTV())
        ub, st = TVImageFiltering.solve(prob, config)
        copyto!(selectdim(expected, 3, b), ub)
        push!(per_stats, st)
    end

    @test isapprox(u_batch, expected; rtol = 0.0, atol = 1e-10)
    @test stats_batch.iterations == maximum(st.iterations for st in per_stats)
    @test stats_batch.converged == all(st.converged for st in per_stats)
    @test stats_batch.rel_change == maximum(st.rel_change for st in per_stats)
end

@testset "solve_batch state reuse and error paths" begin
    Random.seed!(509)
    f_batch = randn(Float64, 8, 8, 3)
    config = TVImageFiltering.ROFConfig(maxiter = 3000, tau = 0.0625, tol = 1e-9, check_every = 10)

    states = [TVImageFiltering.ROFState(selectdim(f_batch, 3, b)) for b = 1:size(f_batch, 3)]
    u1, st1 = TVImageFiltering.solve_batch(
        f_batch,
        config;
        lambda = 0.15,
        tv_mode = TVImageFiltering.AnisotropicTV(),
        state = states,
    )
    u2, st2 = TVImageFiltering.solve_batch(
        f_batch,
        config;
        lambda = 0.15,
        tv_mode = TVImageFiltering.AnisotropicTV(),
        state = states,
        init = fill(3.0, size(f_batch)),
    )

    @test st1.converged
    @test st2.converged
    @test maximum(abs.(u1 .- u2)) <= 1e-5

    @test_throws ArgumentError TVImageFiltering.solve_batch(randn(8), config; lambda = 0.1)
    @test_throws ArgumentError TVImageFiltering.solve_batch(
        f_batch,
        config;
        lambda = 0.1,
        init = zeros(8, 8, 2),
    )
    @test_throws ArgumentError TVImageFiltering.solve_batch(
        f_batch,
        config;
        lambda = 0.1,
        state = states[1:2],
    )
end
