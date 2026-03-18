# Installation

## Use This Repository Directly

From the repository root:

```julia
julia --project=.
```

This activates the package environment in place.

## Use From Another Julia Environment

Develop from a local path:

```julia
import Pkg
Pkg.develop(path="/absolute/path/to/TVImageFiltering.jl")
```

Or add by URL:

```julia
import Pkg
Pkg.add(url="https://github.com/urlicht/TVImageFiltering.jl")
```

## Optional CUDA Support

CUDA support is implemented as an extension (`TVImageFilteringCUDAExt`) and
loads automatically when both packages are available:

```julia
using CUDA
using TVImageFiltering
```

If CUDA is not installed or not functional, CPU functionality remains available.
