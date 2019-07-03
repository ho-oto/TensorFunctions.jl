using Documenter, TensorMaps

makedocs(;
    modules=[TensorMaps],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/ho-oto/TensorMaps.jl/blob/{commit}{path}#L{line}",
    sitename="TensorMaps.jl",
    authors="ho-oto",
    assets=String[],
)

deploydocs(;
    repo="github.com/ho-oto/TensorMaps.jl",
)
