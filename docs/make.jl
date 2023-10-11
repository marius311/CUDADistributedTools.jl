
cd(dirname(@__FILE__))
using Pkg
Pkg.activate(".")

using CUDADistributedTools, Documenter

makedocs(
    sitename="CUDADistributedTools.jl", 
    pages = ["index.md"],
)

deploydocs(
    repo = "github.com/marius311/CUDADistributedTools.jl.git",
    push_preview = true,
    forcepush = true
)
