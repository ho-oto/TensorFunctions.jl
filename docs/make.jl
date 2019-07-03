using Documenter, TensorFunctions

makedocs(;
    modules=[TensorFunctions],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/ho-oto/TensorFunctions.jl/blob/{commit}{path}#L{line}",
    sitename="TensorFunctions.jl",
    authors="ho-oto",
    assets=String[],
)

deploydocs(;
    repo="github.com/ho-oto/TensorFunctions.jl",
)
