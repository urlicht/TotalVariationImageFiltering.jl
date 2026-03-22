# Benchmarking TotalVariationImageFiltering

This benchmark suite is deterministic (fixed seed for synthetic noise) and reusable across machines/commits.

## Setup

```bash
julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'
```

## Run

From the repository root:

```bash
julia --project=benchmark benchmark/run_benchmarks.jl --backend=cpu --samples=20 --output=benchmark/results/cpu.csv
```

To try CUDA too:

```bash
julia --project=benchmark -e 'using Pkg; Pkg.add("CUDA")'
julia --project=benchmark benchmark/run_benchmarks.jl --backend=both --samples=20 --output=benchmark/results/both.csv
```

If CUDA is not installed/functioning (or extension is inactive), CUDA cases are skipped automatically.

## Useful options

- `--backend=cpu|cuda|both`
- `--samples=N`
- `--evals=N`
- `--seed=N`
- `--quick`
- `--output=PATH`
- `--no-output`
- `--no-gpu` (alias for `--backend=cpu`)

Use `--help` for all options:

```bash
julia --project=benchmark benchmark/run_benchmarks.jl --help
```

## CUDA Batching vs Non-Batching

For CUDA-only comparison across batch sizes (`1,4,...,1024`):

```bash
julia --project=benchmark -e 'using Pkg; Pkg.add("CUDA")'
julia --project=benchmark benchmark/compare_batching_cuda.jl \
  --batch-sizes=1,4,8,16,32,64,128,256,512,1024 \
  --output=benchmark/results/cuda_batching_vs_no_batching.csv
```
