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
        "ROF Solver" => "rof-solver.md",
    ],
)

deploydocs(
    repo = "github.com/urlicht/TVImageFiltering.jl.git",
    devbranch = "main",
)
