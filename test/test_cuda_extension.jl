using Test
using Random
using TVImageFiltering

@testset "CUDA extension (optional)" begin
    cuda_loaded = false
    try
        @eval using CUDA
        cuda_loaded = true
    catch
        cuda_loaded = false
    end

    if !cuda_loaded
        @test true
    elseif !CUDA.functional()
        @test true
    else
        @test Base.get_extension(TVImageFiltering, :TVImageFilteringCUDAExt) !== nothing

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
        @test stats_cpu.iterations <= config.maxiter

        f_gpu = CUDA.cu(f_cpu)
        prob_gpu = TVImageFiltering.TVProblem(
            f_gpu;
            lambda = 0.15f0,
            tv_mode = TVImageFiltering.IsotropicTV(),
        )
        u_gpu, stats_gpu = TVImageFiltering.solve(prob_gpu, config)

        @test stats_gpu.iterations <= config.maxiter
        @test isapprox(Array(u_gpu), u_cpu; rtol = 5.0f-4, atol = 5.0f-4)

        f_batch_cpu = rand(Float32, 48, 48, 3)
        u_batch_cpu, stats_batch_cpu =
            TVImageFiltering.solve_batch(f_batch_cpu, config; lambda = 0.15f0)
        f_batch_gpu = CUDA.cu(f_batch_cpu)
        u_batch_gpu, stats_batch_gpu =
            TVImageFiltering.solve_batch(f_batch_gpu, config; lambda = 0.15f0)
        u_sliced_gpu = similar(f_batch_gpu)
        @views for b = 1:size(f_batch_gpu, 3)
            prob_slice = TVImageFiltering.TVProblem(
                selectdim(f_batch_gpu, 3, b);
                lambda = 0.15f0,
                tv_mode = TVImageFiltering.IsotropicTV(),
            )
            u_slice, _ = TVImageFiltering.solve(prob_slice, config)
            copyto!(selectdim(u_sliced_gpu, 3, b), u_slice)
        end

        @test stats_batch_cpu.iterations <= config.maxiter
        @test stats_batch_gpu.iterations <= config.maxiter
        @test isapprox(Array(u_batch_gpu), u_batch_cpu; rtol = 7.0f-4, atol = 7.0f-4)
        @test isapprox(
            Array(u_batch_gpu),
            Array(u_sliced_gpu);
            rtol = 5.0f-5,
            atol = 5.0f-5,
        )

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

        f_batch_pdhg_cpu = rand(Float32, 32, 32, 3)
        f_batch_pdhg_gpu = CUDA.cu(f_batch_pdhg_cpu)
        u_batch_pdhg_cpu, stats_batch_pdhg_cpu = TVImageFiltering.solve_batch(
            f_batch_pdhg_cpu,
            pdhg_config;
            lambda = 0.1f0,
            data_fidelity = TVImageFiltering.L2Fidelity(),
            tv_mode = TVImageFiltering.IsotropicTV(),
        )
        u_batch_pdhg_gpu, stats_batch_pdhg_gpu = TVImageFiltering.solve_batch(
            f_batch_pdhg_gpu,
            pdhg_config;
            lambda = 0.1f0,
            data_fidelity = TVImageFiltering.L2Fidelity(),
            tv_mode = TVImageFiltering.IsotropicTV(),
        )

        @test stats_batch_pdhg_cpu.iterations <= pdhg_config.maxiter
        @test stats_batch_pdhg_gpu.iterations <= pdhg_config.maxiter
        @test isapprox(
            Array(u_batch_pdhg_gpu),
            u_batch_pdhg_cpu;
            rtol = 1.0f-3,
            atol = 1.0f-3,
        )
    end
end
