using Documenter
using TVImageFiltering

const CI = get(ENV, "CI", "false") == "true"

makedocs(
    modules = [TVImageFiltering],
    sitename = "TVImageFiltering.jl",
    checkdocs = :none,
    format = Documenter.HTML(
        prettyurls = CI,
        canonical = "https://urlicht.github.io/TVImageFiltering.jl",
        edit_link = "main",
    ),
    pages = [
        "Home" => "index.md",
        "Installation" => "installation.md",
        "Quick Start" => "quick-start.md",
        "Problem & API" => "problem-and-api.md",
        "ROF Solver" => "rof-solver.md",
        "PDHG Solver" => "pdhg-solver.md",
        "Lambda Selection" => "lambda-selection.md",
        "Batch & CUDA" => "batch-and-cuda.md",
        "Benchmarking" => "benchmarking.md",
        "API Reference" => "api-reference.md",
        "References" => "references.md",
    ],
)

deploydocs(
    repo = "github.com/urlicht/TVImageFiltering.jl.git",
    devbranch = "main",
)
