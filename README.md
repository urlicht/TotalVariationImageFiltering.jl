# TVImageFiltering.jl

[![CI](https://github.com/urlicht/TVImageFiltering.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/urlicht/TVImageFiltering.jl/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://urlicht.github.io/TVImageFiltering.jl/)

`TVImageFiltering.jl` is a Julia package for total-variation (TV) denoising and
reconstruction on `N`-dimensional arrays.

Original implementation: [GPUFilter.jl](https://github.com/flavell-lab/GPUFilter.jl)

![comparison: original, noisy, denoised](docs/img/demo1.png)
![x slice showing a comparison: original, noisy, denoised](docs/img/demo2.png)

## Documentation

Full manual: [https://urlicht.github.io/TVImageFiltering.jl/](https://urlicht.github.io/TVImageFiltering.jl/)

Recommended entry points:

- [Quick Start](https://urlicht.github.io/TVImageFiltering.jl/quick-start/)
- [Problem & API](https://urlicht.github.io/TVImageFiltering.jl/problem-and-api/)
- [ROF Solver](https://urlicht.github.io/TVImageFiltering.jl/rof-solver/)
- [PDHG Solver](https://urlicht.github.io/TVImageFiltering.jl/pdhg-solver/)
- [Lambda Selection](https://urlicht.github.io/TVImageFiltering.jl/lambda-selection/)
- [Batch & CUDA](https://urlicht.github.io/TVImageFiltering.jl/batch-and-cuda/)
- [API Reference](https://urlicht.github.io/TVImageFiltering.jl/api-reference/)

## Features

- ROF denoising (`L2 + TV`) with Chambolle dual projection
- PDHG / Chambolle-Pock for `L2 + TV` and Poisson `KL + TV`
- Isotropic and anisotropic TV
- Single-image and batched solves
- Automatic lambda selection for ROF (discrepancy principle and MC-SURE)
- Optional CUDA acceleration via package extension

## Installation

From this repository:

```julia
julia --project=.
```

From another Julia environment (local path):

```julia
import Pkg
Pkg.develop(path="/absolute/path/to/TVImageFiltering.jl")
```

From a hosted repository:

```julia
import Pkg
Pkg.add(url="https://github.com/urlicht/TVImageFiltering.jl")
```

## Minimal Example

```julia
using TVImageFiltering

f = rand(Float32, 128, 128)
problem = TVImageFiltering.TVProblem(
    f;
    lambda = 0.1f0,
    data_fidelity = TVImageFiltering.L2Fidelity(),
    tv_mode = TVImageFiltering.IsotropicTV(),
)

u, stats = TVImageFiltering.solve(problem, TVImageFiltering.ROFConfig())
```

## Benchmarking

Benchmark instructions and scripts are documented in
[benchmark/README.md](benchmark/README.md).

## Testing

```julia
import Pkg
Pkg.test()
```

CUDA tests run only when CUDA is installed and functional.
