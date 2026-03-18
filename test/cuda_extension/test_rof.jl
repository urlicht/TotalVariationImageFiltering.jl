using Test
using Random
using TVImageFiltering
using CUDA

@testset "CUDA ROF Single-Image Parity" begin
    Random.seed!(211)
    f_cpu = rand(Float32, 64, 64)
    config = TVImageFiltering.ROFConfig(
        maxiter = 400,
        tau = 0.0625f0,
        tol = 1.0f-5,
        check_every = 10,
    )

    prob_cpu = TVImageFiltering.TVProblem(
        f_cpu;
        lambda = 0.15f0,
        tv_mode = TVImageFiltering.IsotropicTV(),
    )
    u_cpu, stats_cpu = TVImageFiltering.solve(prob_cpu, config)

    f_gpu = CUDA.cu(f_cpu)
    prob_gpu = TVImageFiltering.TVProblem(
        f_gpu;
        lambda = 0.15f0,
        tv_mode = TVImageFiltering.IsotropicTV(),
    )
    u_gpu, stats_gpu = TVImageFiltering.solve(prob_gpu, config)

    @test stats_cpu.iterations <= config.maxiter
    @test stats_gpu.iterations <= config.maxiter
    @test isapprox(Array(u_gpu), u_cpu; rtol = 5.0f-4, atol = 5.0f-4)
end
