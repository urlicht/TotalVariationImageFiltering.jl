using Test
using Random
using TotalVariationImageFiltering

@testset "allocate_dual" begin
    u = randn(Float32, 2, 3, 4)
    p = TotalVariationImageFiltering.allocate_dual(u)

    @test length(p) == 3
    @test all(size(pd) == size(u) for pd in p)
    @test all(eltype(pd) == Float32 for pd in p)
    @test p[1] !== p[2]
    @test p[2] !== p[3]

    fill!(p[1], 1.0f0)
    @test !all(p[2] .== 1.0f0)
end

@testset "gradient! exact values" begin
    u1 = [1.0, 4.0, 10.0]
    g1 = TotalVariationImageFiltering.allocate_dual(u1)

    TotalVariationImageFiltering.gradient!(g1, u1, TotalVariationImageFiltering.Neumann())
    @test g1[1] == [3.0, 6.0, 0.0]

    TotalVariationImageFiltering.gradient!(g1, u1, TotalVariationImageFiltering.Neumann(), (2.0,))
    @test g1[1] == [6.0, 12.0, 0.0]

    u2 = [1.0 2.0 3.0; 4.0 5.0 6.0]
    g2 = TotalVariationImageFiltering.allocate_dual(u2)
    TotalVariationImageFiltering.gradient!(g2, u2, TotalVariationImageFiltering.Neumann())
    @test g2[1] == [3.0 3.0 3.0; 0.0 0.0 0.0]
    @test g2[2] == [1.0 1.0 0.0; 1.0 1.0 0.0]

    TotalVariationImageFiltering.gradient!(g2, u2, TotalVariationImageFiltering.Neumann(), (0.5, 2.0))
    @test g2[1] == [1.5 1.5 1.5; 0.0 0.0 0.0]
    @test g2[2] == [2.0 2.0 0.0; 2.0 2.0 0.0]
end

@testset "divergence! exact values" begin
    out = zeros(3)
    p = ([2.0, -1.0, 7.5],)

    TotalVariationImageFiltering.divergence!(out, p, TotalVariationImageFiltering.Neumann())
    @test out == [2.0, -3.0, 1.0]

    TotalVariationImageFiltering.divergence!(out, p, TotalVariationImageFiltering.Neumann(), (2.0,))
    @test out == [4.0, -6.0, 2.0]

    out_singleton = fill(42.0, 1)
    p_singleton = ([3.0],)
    TotalVariationImageFiltering.divergence!(out_singleton, p_singleton, TotalVariationImageFiltering.Neumann())
    @test out_singleton == [0.0]
end

@testset "Scaled adjointness of gradient/divergence" begin
    Random.seed!(23)
    for sz in ((7,), (3, 4), (2, 3, 4), (1, 5, 1))
        u = randn(Float64, sz...)
        n_dims = ndims(u)
        spacing = ntuple(d -> 0.5 + d, n_dims)
        inv_spacing = ntuple(d -> inv(spacing[d]), n_dims)
        p = ntuple(_ -> randn(Float64, sz...), n_dims)
        g = TotalVariationImageFiltering.allocate_dual(u)
        divp = similar(u)

        TotalVariationImageFiltering.gradient!(g, u, TotalVariationImageFiltering.Neumann(), inv_spacing)
        TotalVariationImageFiltering.divergence!(divp, p, TotalVariationImageFiltering.Neumann(), inv_spacing)

        lhs = sum(sum(g[d] .* p[d]) for d = 1:n_dims)
        rhs = sum(u .* divp)
        @test isapprox(lhs + rhs, 0.0; atol = 1e-10, rtol = 0.0)
    end
end

@testset "project_dual_ball! isotropic and anisotropic" begin
    p_iso = ([3.0, 0.6, -0.3], [4.0, 0.8, 0.4])
    TotalVariationImageFiltering.project_dual_ball!(p_iso, 1.0, TotalVariationImageFiltering.IsotropicTV())
    @test isapprox(p_iso[1][1], 0.6; atol = 1e-12)
    @test isapprox(p_iso[2][1], 0.8; atol = 1e-12)
    @test p_iso[1][2] == 0.6
    @test p_iso[2][2] == 0.8
    @test p_iso[1][3] == -0.3
    @test p_iso[2][3] == 0.4

    p_aniso = ([-2.0, -0.2, 0.3, 1.7], [0.5, -3.1, 2.2, -0.9])
    TotalVariationImageFiltering.project_dual_ball!(p_aniso, 1.0, TotalVariationImageFiltering.AnisotropicTV())
    @test p_aniso[1] == [-1.0, -0.2, 0.3, 1.0]
    @test p_aniso[2] == [0.5, -1.0, 1.0, -0.9]
end
