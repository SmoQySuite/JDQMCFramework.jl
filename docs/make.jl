using JDQMCFramework
using Documenter
using Literate
using LinearAlgebra

tutorial_names = ["square_hubbard",]
tutorial_literate_sources = [joinpath(@__DIR__, "..", "literate_scripts", name*".jl") for name in tutorial_names]
tutorial_script_destinations = [joinpath(@__DIR__, "..", "tutorial_scripts") for name in tutorial_names]
tutorial_notebook_destinations = [joinpath(@__DIR__, "..", "tutorial_notebooks") for name in tutorial_names]
tutorial_documentation_destination = joinpath(@__DIR__, "src", "tutorials")
tutorial_documentation_paths = ["tutorials/$name.md" for name in tutorial_names]

DocMeta.setdocmeta!(JDQMCFramework, :DocTestSetup, :(using JDQMCFramework); recursive=true)

for i in eachindex(tutorial_names)
    Literate.markdown(
        tutorial_literate_sources[i],
        tutorial_documentation_destination; 
        execute = true,
        documenter = true
    )
    Literate.script(
        tutorial_literate_sources[i],
        tutorial_script_destinations[i]
    )
    Literate.notebook(
        tutorial_literate_sources[i],
        tutorial_notebook_destinations[i],
        execute = false
    )
end

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
        "API" => "api.md",
        "Tutorials" => tutorial_documentation_paths,
    ],
)

deploydocs(;
    repo="github.com/SmoQySuite/JDQMCFramework.jl.git",
    devbranch="master",
)
