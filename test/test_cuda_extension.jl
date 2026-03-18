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
    end
end
