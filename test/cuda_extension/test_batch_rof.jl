using Test
using Random
using TVImageFiltering
using CUDA

@testset "CUDA ROF Batch API" begin
    Random.seed!(223)
    f_batch_cpu = rand(Float32, 48, 48, 3)
    config = TVImageFiltering.ROFConfig(
        maxiter = 400,
        tau = 0.0625f0,
        tol = 1.0f-5,
        check_every = 10,
    )

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
    @test isapprox(Array(u_batch_gpu), Array(u_sliced_gpu); rtol = 5.0f-5, atol = 5.0f-5)
end
