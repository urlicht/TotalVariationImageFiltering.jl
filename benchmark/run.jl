#!/usr/bin/env julia

import Pkg

Pkg.activate(@__DIR__)

using BenchmarkTools
using Dates
using Printf
using Random
using TestImages
using TVImageFiltering

const _USAGE = """
Usage:
  julia --project=benchmark benchmark/run.jl [--quick] [--no-gpu]

Options:
  --quick    Run fewer/smaller benchmark cases.
  --no-gpu   Skip CUDA benchmarks even if CUDA.jl is available.
  --help     Show this help text.
"""

function parse_args(args::Vector{String})
    quick = false
    run_gpu = true

    for arg in args
        if arg == "--quick"
            quick = true
        elseif arg == "--no-gpu"
            run_gpu = false
        elseif arg == "--help"
            println(_USAGE)
            exit(0)
        else
            error("Unknown argument: $(arg)\n\n$(_USAGE)")
        end
    end

    return (quick = quick, run_gpu = run_gpu)
end

function configure_benchmark_defaults!(; quick::Bool)
    BenchmarkTools.DEFAULT_PARAMETERS.evals = 1
    BenchmarkTools.DEFAULT_PARAMETERS.samples = quick ? 8 : 20
    BenchmarkTools.DEFAULT_PARAMETERS.seconds = quick ? 1.0 : 3.0
    BenchmarkTools.DEFAULT_PARAMETERS.gctrial = true
    BenchmarkTools.DEFAULT_PARAMETERS.gcsample = false
    return nothing
end

function benchmark_cases(; quick::Bool)
    quick && return [
        (
            name = "cameraman-gaussian",
            image = "cameraman",
            mode = TVImageFiltering.IsotropicTV(),
            lambda = 0.12f0,
            gaussian_sigma = 0.04f0,
            salt_pepper_prob = 0.00f0,
            seed = 101,
            gpu = true,
        ),
        (
            name = "pirate-gaussian",
            image = "pirate",
            mode = TVImageFiltering.AnisotropicTV(),
            lambda = 0.16f0,
            gaussian_sigma = 0.06f0,
            salt_pepper_prob = 0.00f0,
            seed = 103,
            gpu = true,
        ),
    ]

    return [
        (
            name = "cameraman-gaussian",
            image = "cameraman",
            mode = TVImageFiltering.IsotropicTV(),
            lambda = 0.12f0,
            gaussian_sigma = 0.04f0,
            salt_pepper_prob = 0.00f0,
            seed = 101,
            gpu = true,
        ),
        (
            name = "pirate-gaussian",
            image = "pirate",
            mode = TVImageFiltering.AnisotropicTV(),
            lambda = 0.16f0,
            gaussian_sigma = 0.06f0,
            salt_pepper_prob = 0.00f0,
            seed = 103,
            gpu = true,
        ),
        (
            name = "woman_blonde-mixed",
            image = "woman_blonde",
            mode = TVImageFiltering.IsotropicTV(),
            lambda = 0.14f0,
            gaussian_sigma = 0.03f0,
            salt_pepper_prob = 0.02f0,
            seed = 107,
            gpu = true,
        ),
        (
            name = "mri-stack-gaussian",
            image = "mri-stack",
            mode = TVImageFiltering.IsotropicTV(),
            lambda = 0.10f0,
            gaussian_sigma = 0.02f0,
            salt_pepper_prob = 0.00f0,
            seed = 109,
            gpu = false,
        ),
        (
            name = "resolution_test_1920-large",
            image = "resolution_test_1920",
            mode = TVImageFiltering.IsotropicTV(),
            lambda = 0.10f0,
            gaussian_sigma = 0.02f0,
            salt_pepper_prob = 0.00f0,
            seed = 113,
            gpu = false,
        ),
    ]
end

function normalize01!(data::AbstractArray{Float32})
    lo, hi = extrema(data)
    if hi > lo
        scale = inv(hi - lo)
        @inbounds for i in eachindex(data)
            data[i] = (data[i] - lo) * scale
        end
    else
        fill!(data, 0f0)
    end
    return data
end

function maybe_downsample(data::AbstractArray{Float32}; quick::Bool)
    quick || return data
    idx = ntuple(d -> 1:2:size(data, d), ndims(data))
    return copy(view(data, idx...))
end

function add_noise(
    clean::AbstractArray{Float32};
    seed::Int,
    gaussian_sigma::Float32,
    salt_pepper_prob::Float32,
)
    rng = MersenneTwister(seed)
    noisy = copy(clean)

    @inbounds for i in eachindex(noisy)
        noisy[i] = clamp(noisy[i] + gaussian_sigma * randn(rng, Float32), 0f0, 1f0)
    end

    if salt_pepper_prob > 0f0
        half = salt_pepper_prob / 2f0
        upper = 1f0 - half
        @inbounds for i in eachindex(noisy)
            r = rand(rng, Float32)
            if r < half
                noisy[i] = 0f0
            elseif r > upper
                noisy[i] = 1f0
            end
        end
    end

    return noisy
end

function load_testimage_case(case; quick::Bool)
    raw = TestImages.testimage(case.image)
    clean = Array(Float32.(raw))
    normalize01!(clean)
    clean = maybe_downsample(clean; quick = quick)

    noisy = add_noise(
        clean;
        seed = case.seed,
        gaussian_sigma = case.gaussian_sigma,
        salt_pepper_prob = case.salt_pepper_prob,
    )

    return (
        name = case.name,
        image = case.image,
        mode = case.mode,
        lambda = case.lambda,
        gaussian_sigma = case.gaussian_sigma,
        salt_pepper_prob = case.salt_pepper_prob,
        gpu = case.gpu,
        noisy = noisy,
    )
end

function prepare_cases(; quick::Bool)
    prepared = NamedTuple[]

    for case in benchmark_cases(; quick = quick)
        loaded = try
            load_testimage_case(case; quick = quick)
        catch err
            @warn "Skipping benchmark case because image loading failed." case = case.name image =
                case.image error = err
            continue
        end

        @info "Prepared benchmark case." case = loaded.name image = loaded.image size =
            size(loaded.noisy) gaussian_sigma = loaded.gaussian_sigma salt_pepper_prob =
            loaded.salt_pepper_prob
        push!(prepared, loaded)
    end

    isempty(prepared) && error(
        "No benchmark cases could be prepared from TestImages.jl. Ensure images are available/downloadable.",
    )

    return prepared
end

function build_cpu_suite!(suite::BenchmarkGroup, cases; quick::Bool)
    suite["cpu"] = BenchmarkGroup()
    cpu_group = suite["cpu"]

    maxiter = quick ? 120 : 240

    for case in cases
        problem = TVImageFiltering.TVProblem(
            case.noisy;
            lambda = case.lambda,
            tv_mode = case.mode,
        )
        config = TVImageFiltering.ROFConfig(
            maxiter = maxiter,
            tau = 0.1f0,
            tol = 0.0f0,
            check_every = maxiter,
        )

        subgroup = BenchmarkGroup()
        cpu_group[case.name] = subgroup

        u = similar(case.noisy)
        state = TVImageFiltering.ROFState(case.noisy)

        subgroup["solve (allocating)"] =
            @benchmarkable TVImageFiltering.solve($problem, $config)
        subgroup["solve! (state reuse)"] = @benchmarkable TVImageFiltering.solve!(
            $u,
            $problem,
            $config;
            state = $state,
        ) setup = (copyto!($u, $(case.noisy)))
    end

    return nothing
end

function maybe_load_cuda()
    cuda_mod = try
        @eval using CUDA
        getfield(Main, :CUDA)
    catch err
        @info "CUDA.jl not available; skipping GPU benchmarks." error = err
        return nothing
    end

    functional = try
        Base.invokelatest(getproperty(cuda_mod, :functional))
    catch err
        @info "CUDA loaded but functional() failed; skipping GPU benchmarks." error = err
        return nothing
    end

    if !functional
        @info "CUDA is installed but no functional device was detected; skipping GPU benchmarks."
        return nothing
    end

    return cuda_mod
end

function build_gpu_suite!(suite::BenchmarkGroup, cases, cuda_mod::Module; quick::Bool)
    suite["gpu"] = BenchmarkGroup()
    gpu_group = suite["gpu"]

    maxiter = quick ? 120 : 240
    gpu_cases = filter(case -> ndims(case.noisy) == 2 && case.gpu, cases)
    cu = getproperty(cuda_mod, :cu)
    synchronize = getproperty(cuda_mod, :synchronize)

    for case in gpu_cases
        noisy_gpu = Base.invokelatest(cu, case.noisy)

        problem = TVImageFiltering.TVProblem(
            noisy_gpu;
            lambda = case.lambda,
            tv_mode = case.mode,
        )
        config = TVImageFiltering.ROFConfig(
            maxiter = maxiter,
            tau = 0.1f0,
            tol = 0.0f0,
            check_every = maxiter,
        )

        subgroup = BenchmarkGroup()
        gpu_group[case.name] = subgroup

        u_gpu = similar(noisy_gpu)
        state = TVImageFiltering.ROFState(noisy_gpu)

        subgroup["solve (allocating)"] = @benchmarkable begin
            TVImageFiltering.solve($problem, $config)
            Base.invokelatest($synchronize)
        end
        subgroup["solve! (state reuse)"] =
            @benchmarkable begin
                TVImageFiltering.solve!(
                $u_gpu,
                $problem,
                $config;
                state = $state,
            )
                Base.invokelatest($synchronize)
            end setup = (copyto!($u_gpu, $noisy_gpu); Base.invokelatest($synchronize))
    end

    return nothing
end

function collect_rows(group::BenchmarkGroup, prefix::Vector{String} = String[])
    rows = NamedTuple{(:name, :time_ns, :memory_bytes, :allocs),Tuple{String,Float64,Int,Int}}[]

    for key in sort!(collect(keys(group)); by = x -> string(x))
        value = group[key]
        path = [prefix...; string(key)]

        if value isa BenchmarkGroup
            append!(rows, collect_rows(value, path))
        elseif value isa BenchmarkTools.Trial
            tmin = minimum(value).time
            push!(
                rows,
                (
                    name = join(path, " / "),
                    time_ns = float(tmin),
                    memory_bytes = value.memory,
                    allocs = value.allocs,
                ),
            )
        end
    end

    return rows
end

function format_bytes(n::Integer)
    units = ("B", "KiB", "MiB", "GiB")
    value = float(n)
    unit_idx = 1
    while value >= 1024 && unit_idx < length(units)
        value /= 1024
        unit_idx += 1
    end
    return @sprintf("%.2f %s", value, units[unit_idx])
end

function print_summary(results::BenchmarkGroup)
    rows = collect_rows(results)
    isempty(rows) && return nothing

    println("\nSummary (minimum sample time)")
    @printf("%-52s %12s %13s %10s\n", "Benchmark", "Time", "Memory", "Allocs")
    @printf("%s\n", repeat("-", 92))
    for row in rows
        @printf(
            "%-52s %10.3f ms %13s %10d\n",
            row.name,
            row.time_ns / 1.0e6,
            format_bytes(row.memory_bytes),
            row.allocs,
        )
    end
    return nothing
end

function csv_escape(value)
    text = string(value)
    if occursin(',', text) || occursin('"', text) || occursin('\n', text)
        return "\"" * replace(text, "\"" => "\"\"") * "\""
    end
    return text
end

function write_results_csv(results::BenchmarkGroup; quick::Bool, gpu_requested::Bool)
    rows = collect_rows(results)
    isempty(rows) && return nothing

    root_dir = normpath(joinpath(@__DIR__, ".."))
    results_dir = joinpath(root_dir, "results")
    mkpath(results_dir)
    csv_path = joinpath(results_dir, "benchmark_results.csv")
    timestamp = Dates.format(Dates.now(Dates.UTC), dateformat"yyyy-mm-ddTHH:MM:SS")

    open(csv_path, "w") do io
        println(
            io,
            "run_timestamp_utc,benchmark,time_ns,time_ms,memory_bytes,allocs,quick_mode,gpu_requested",
        )
        for row in rows
            cols = (
                timestamp,
                row.name,
                @sprintf("%.0f", row.time_ns),
                @sprintf("%.6f", row.time_ns / 1.0e6),
                string(row.memory_bytes),
                string(row.allocs),
                string(quick),
                string(gpu_requested),
            )
            println(io, join(csv_escape.(cols), ","))
        end
    end

    return csv_path
end

function main(args::Vector{String})
    opts = parse_args(args)
    configure_benchmark_defaults!(; quick = opts.quick)

    println("Preparing TestImages-based cases...")
    cases = prepare_cases(; quick = opts.quick)

    suite = BenchmarkGroup()
    build_cpu_suite!(suite, cases; quick = opts.quick)

    if opts.run_gpu
        cuda_mod = maybe_load_cuda()
        cuda_mod === nothing || build_gpu_suite!(suite, cases, cuda_mod; quick = opts.quick)
    end

    println("Running TVImageFiltering benchmarks...")
    println("  quick mode: $(opts.quick)")
    println("  gpu enabled: $(opts.run_gpu)")

    results = run(suite; verbose = true)
    print_summary(results)
    csv_path = write_results_csv(results; quick = opts.quick, gpu_requested = opts.run_gpu)
    csv_path === nothing || println("\nWrote CSV results to: $(csv_path)")
    return nothing
end

main(ARGS)
