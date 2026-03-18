using Test
using Random
using TVImageFiltering
using CUDA

@testset "CUDA PDHG L2 and Poisson Parity" begin
    Random.seed!(233)
    f_cpu = rand(Float32, 64, 64)
    f_gpu = CUDA.cu(f_cpu)

    pdhg_config = TVImageFiltering.PDHGConfig(
        maxiter = 1500,
        tau = 0.2f0,
        sigma = 0.2f0,
        theta = 1.0f0,
        tol = 1.0f-5,
        check_every = 10,
    )

    prob_cpu_pdhg_l2 = TVImageFiltering.TVProblem(
        f_cpu;
        lambda = 0.15f0,
        data_fidelity = TVImageFiltering.L2Fidelity(),
        tv_mode = TVImageFiltering.IsotropicTV(),
    )
    u_cpu_pdhg_l2, stats_cpu_pdhg_l2 = TVImageFiltering.solve(prob_cpu_pdhg_l2, pdhg_config)
    prob_gpu_pdhg_l2 = TVImageFiltering.TVProblem(
        f_gpu;
        lambda = 0.15f0,
        data_fidelity = TVImageFiltering.L2Fidelity(),
        tv_mode = TVImageFiltering.IsotropicTV(),
    )
    u_gpu_pdhg_l2, stats_gpu_pdhg_l2 = TVImageFiltering.solve(prob_gpu_pdhg_l2, pdhg_config)

    @test stats_cpu_pdhg_l2.iterations <= pdhg_config.maxiter
    @test stats_gpu_pdhg_l2.iterations <= pdhg_config.maxiter
    @test isapprox(Array(u_gpu_pdhg_l2), u_cpu_pdhg_l2; rtol = 8.0f-4, atol = 8.0f-4)

    f_poisson_cpu = abs.(f_cpu) .+ 0.1f0
    f_poisson_gpu = CUDA.cu(f_poisson_cpu)
    prob_cpu_pdhg_poisson = TVImageFiltering.TVProblem(
        f_poisson_cpu;
        lambda = 0.12f0,
        data_fidelity = TVImageFiltering.PoissonFidelity(),
        tv_mode = TVImageFiltering.IsotropicTV(),
    )
    u_cpu_pdhg_poisson, stats_cpu_pdhg_poisson =
        TVImageFiltering.solve(prob_cpu_pdhg_poisson, pdhg_config)
    prob_gpu_pdhg_poisson = TVImageFiltering.TVProblem(
        f_poisson_gpu;
        lambda = 0.12f0,
        data_fidelity = TVImageFiltering.PoissonFidelity(),
        tv_mode = TVImageFiltering.IsotropicTV(),
    )
    u_gpu_pdhg_poisson, stats_gpu_pdhg_poisson =
        TVImageFiltering.solve(prob_gpu_pdhg_poisson, pdhg_config)

    @test stats_cpu_pdhg_poisson.iterations <= pdhg_config.maxiter
    @test stats_gpu_pdhg_poisson.iterations <= pdhg_config.maxiter
    @test minimum(Array(u_gpu_pdhg_poisson)) >= -1.0f-5
    @test isapprox(
        Array(u_gpu_pdhg_poisson),
        u_cpu_pdhg_poisson;
        rtol = 1.5f-3,
        atol = 1.5f-3,
    )
end
