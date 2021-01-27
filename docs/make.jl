include(joinpath(dirname(@__DIR__), "src", "FaceDetection.jl"))
using Documenter, .FaceDetection

Documenter.makedocs(
    clean = true,
    doctest = true,
    modules = Module[FaceDetection],
    repo = "",
    highlightsig = true,
    sitename = "FaceDetection Documentation",
    expandfirst = [],
    pages = [
        "Home" => "index.md",
        "Usage" => "usage.md",
        "Examples" => "examples.md",
        "Benchmarking Results" => "benchmarking.md",
        "Caveats" => "caveats.md",
        "Other Resources" => "resources.md",
        "A Few Acknowledgements" => "acknowledgements.md"
    ]
)

deploydocs(;
    repo  =  "github.com/jakewilliami/FaceDetection.jl.git",
)
