# Benchmarking

This section summarizes the benchmark tooling in `benchmark/README.md` and
benchmark scripts.

## Setup

```bash
julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'
```

## Main Benchmark Script

Run from repository root:

```bash
julia --project=benchmark benchmark/run_benchmarks.jl --backend=cpu --samples=20 --output=benchmark/results/cpu.csv
```

Run CPU and CUDA together:

```bash
julia --project=benchmark -e 'using Pkg; Pkg.add("CUDA")'
julia --project=benchmark benchmark/run_benchmarks.jl --backend=both --samples=20 --output=benchmark/results/both.csv
```

If CUDA is missing/inactive, CUDA cases are skipped automatically.

## Useful Options

- `--backend=cpu|cuda|both`
- `--samples=N`
- `--evals=N`
- `--seed=N`
- `--quick`
- `--output=PATH`
- `--no-output`
- `--no-gpu` (alias for `--backend=cpu`)

Help:

```bash
julia --project=benchmark benchmark/run_benchmarks.jl --help
```

## CUDA Batching vs Non-Batching

Dedicated comparison script:

```bash
julia --project=benchmark -e 'using Pkg; Pkg.add("CUDA")'
julia --project=benchmark benchmark/compare_batching_cuda.jl \
  --batch-sizes=1,4,8,16,32,64,128,256,512,1024 \
  --output=benchmark/results/cuda_batching_vs_no_batching.csv
```

This benchmark compares:

- looping over many single-image solves (`no_batch`),
- one batched solve (`batched`).

## Benchmark Cases and Data

The main script benchmarks denoising on TestImages-based cases including:

- `cameraman`
- `pirate`
- `woman_blonde`
- `mri-stack`
- `resolution_test_1920`

with controlled synthetic noise settings and deterministic RNG seeds.

## Output

Results are written as CSV files in `benchmark/results/`, including timing
statistics (`median/mean/min`), memory, allocations, and run metadata.
