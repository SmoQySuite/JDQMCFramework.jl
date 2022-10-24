using JDQMCFramework
using Documenter
using LinearAlgebra

DocMeta.setdocmeta!(JDQMCFramework, :DocTestSetup, :(using JDQMCFramework); recursive=true)

makedocs(;
    modules=[JDQMCFramework],
    authors="Benjamin Cohen-Stead <benwcs@gmail.com>",
    repo="https://github.com/SmoQySuite/JDQMCFramework.jl/blob/{commit}{path}#{line}",
    sitename="JDQMCFramework.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://SmoQySuite.github.io/JDQMCFramework.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "API" => "api.md"
    ],
)

deploydocs(;
    repo="github.com/SmoQySuite/JDQMCFramework.jl",
    devbranch="master",
)
