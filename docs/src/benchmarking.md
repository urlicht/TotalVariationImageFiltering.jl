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

## Benchmark Results Snapshot

This is the benchmark result block previously shown in `README.md`.
Times are in milliseconds and are hardware-dependent.

### Command

```bash
julia --project=benchmark benchmark/run_benchmarks.jl --backend=both --samples=10 --output=benchmark/results/both.csv
```

### Results

| Backend | Case | Image | Dims | Median | Mean | Min | Memory | Allocs |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| cpu | solve_allocating | cameraman | 512x512 | 692.518 | 700.534 | 688.055 | 12775280 | 45630 |
| cpu | solve_state_reuse | cameraman | 512x512 | 749.794 | 745.226 | 721.041 | 3337304 | 45603 |
| cpu | solve_allocating | pirate | 512x512 | 628.141 | 631.987 | 618.272 | 12775280 | 45630 |
| cpu | solve_state_reuse | pirate | 512x512 | 622.439 | 622.986 | 613.514 | 3337304 | 45603 |
| cpu | solve_allocating | woman_blonde | 512x512 | 745.798 | 749.035 | 740.033 | 12775280 | 45630 |
| cpu | solve_state_reuse | woman_blonde | 512x512 | 816.535 | 813.798 | 805.887 | 3337304 | 45603 |
| cpu | solve_allocating | mri-stack | 226x186x27 | 4702.706 | 4702.706 | 4677.952 | 58350624 | 74196 |
| cpu | solve_state_reuse | mri-stack | 226x186x27 | 4869.636 | 4869.636 | 4809.036 | 8410712 | 74163 |
| cpu | solve_allocating | resolution_test_1920 | 1920x1920 | 11679.991 | 11679.991 | 11679.991 | 149745520 | 45630 |
| cpu | solve_state_reuse | resolution_test_1920 | 1920x1920 | 11514.213 | 11514.213 | 11514.213 | 17034328 | 45603 |
| cuda | solve_allocating | cameraman | 512x512 | 27.272 | 27.476 | 26.933 | 1933344 | 54328 |
| cuda | solve_state_reuse | cameraman | 512x512 | 27.182 | 27.174 | 26.949 | 1992752 | 54492 |
| cuda | solve_allocating | pirate | 512x512 | 27.768 | 27.743 | 27.394 | 1934048 | 54342 |
| cuda | solve_state_reuse | pirate | 512x512 | 27.476 | 27.982 | 27.389 | 1931168 | 54243 |
| cuda | solve_allocating | woman_blonde | 512x512 | 27.156 | 27.373 | 26.851 | 1995312 | 54571 |
| cuda | solve_state_reuse | woman_blonde | 512x512 | 27.439 | 27.441 | 27.347 | 1992624 | 54484 |
| cuda | solve_allocating | mri-stack | 226x186x27 | 194.974 | 194.945 | 194.536 | 2760464 | 65412 |
| cuda | solve_state_reuse | mri-stack | 226x186x27 | 197.457 | 197.412 | 196.523 | 2756768 | 65291 |
| cuda | solve_allocating | resolution_test_1920 | 1920x1920 | 384.540 | 384.467 | 383.904 | 1934560 | 54404 |
| cuda | solve_state_reuse | resolution_test_1920 | 1920x1920 | 384.019 | 383.912 | 383.151 | 1931872 | 54317 |

### Environment

```text
CUDA toolchain:
- runtime 13.2, artifact installation
- driver 580.95.5 for 13.2
- compiler 13.2

CUDA libraries:
- CUBLAS: 13.1.0
- CURAND: 10.4.2
- CUFFT: 12.2.0
- CUSOLVER: 12.1.0
- CUSPARSE: 12.7.9
- CUPTI: 2026.1.0 (API 13.2.0)
- NVML: 13.0.0+580.95.5

Julia packages:
- CUDA: 5.11.0
- GPUArrays: 11.4.1
- GPUCompiler: 1.8.2
- KernelAbstractions: 0.9.40
- CUDA_Driver_jll: 13.2.0+0
- CUDA_Compiler_jll: 0.4.2+0
- CUDA_Runtime_jll: 0.21.0+0

Toolchain:
- Julia: 1.12.5
- LLVM: 18.1.7

1 device:
  0: NVIDIA GeForce RTX 3060 (sm_86, 11.626 GiB / 12.000 GiB available)
```
