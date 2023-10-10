# ensure in right directory and environment
cd(dirname(@__FILE__))
using Pkg
pkg"activate ."

using CUDADistributedTools, Documenter

makedocs(
    sitename="CUDADistributedTools.jl", 
    pages = [
        "index.md",
    ],
    remotes = nothing
)

deploydocs(
    repo = "github.com/marius311/CUDADistributedTools.jl.git",
    push_preview = true,
    forcepush = true
)
