using Test
using TVImageFiltering

@testset "CUDA extension (optional)" begin
    cuda_loaded = false
    try
        @eval using CUDA
        cuda_loaded = true
    catch
        cuda_loaded = false
    end

    if !cuda_loaded || !CUDA.functional()
        @test true
    else
        @test Base.get_extension(TVImageFiltering, :TVImageFilteringCUDAExt) !== nothing
        include("cuda_extension/test_rof.jl")
        include("cuda_extension/test_batch_rof.jl")
        include("cuda_extension/test_pdhg_l2_poisson.jl")
        include("cuda_extension/test_constraints.jl")
        include("cuda_extension/test_pdhg_batch.jl")
    end
end
