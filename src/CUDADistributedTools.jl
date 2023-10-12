module CUDADistributedTools

using Distributed


"""
    assign_GPU_workers(;
        print_info = true, 
        remove_oversubscribed_workers = false
        master_has_own_gpu = false, 
    )

Assign each Julia process a unique GPU. Assignment is done using `CUDA.device!`.
Works with workers which may be distributed across different hosts,
and with each host possibly having multiple GPUs.

# Keyword arguments

* `remove_oversubscribed_workers` — If `true`, remove worker processes for which 
  there are no free GPUs left. If `false`, an error is raised instead in this case.

* `master_has_own_gpu` — Controls whether the master process counts as using 
  its GPU, or whether a worker is free to also be assigned the same GPU.

* `print_info` - Print [`CUDADistributedTools.proc_info`](@ref) after assignment.

# Example

```julia-repl
julia> using Distributed, CUDA, CUDADistributedTools

julia> CUDA.devices()
CUDA.DeviceIterator() for 4 devices:
0. NVIDIA A100-SXM4-40GB
1. NVIDIA A100-SXM4-40GB
2. NVIDIA A100-SXM4-40GB
3. NVIDIA A100-SXM4-40GB

julia> addprocs(3)
3-element Vector{Int64}:
 2
 3
 4

julia> assign_GPU_workers()
┌ Info: Processes (4):
│  (myid = 1, host = nid001293, device = CuDevice(0): NVIDIA A100-SXM4-40GB 1c40175b))
│  (myid = 2, host = nid001293, device = CuDevice(1): NVIDIA A100-SXM4-40GB f179efe2))
│  (myid = 3, host = nid001293, device = CuDevice(2): NVIDIA A100-SXM4-40GB 36d32866))
└  (myid = 4, host = nid001293, device = CuDevice(3): NVIDIA A100-SXM4-40GB 634451b9))
```

"""
function assign_GPU_workers(;print_info=true, master_has_own_gpu=false, remove_oversubscribed_workers=false)
    if nprocs() > 1
        @everywhere @eval Main using Distributed, CUDA, CUDADistributedTools
        master_uuid = @eval Main CUDA.uuid(device())
        accessible_gpus = Dict(asyncmap(workers()) do id
            @eval Main @fetchfrom $id begin
                ds = CUDA.devices()
                # put master's GPU last so we don't double up on it unless we need to
                $id => sort((CUDA.deviceid.(ds) .=> CUDA.uuid.(ds)), by=(((k,v),)->v==$master_uuid ? Inf : k))
            end
        end)
        claimed = master_has_own_gpu ? Set([master_uuid]) : Set()
        assignments = Dict(map(workers()) do myid
            for (gpu_id, gpu_uuid) in accessible_gpus[myid]
                if !(gpu_uuid in claimed)
                    push!(claimed, gpu_uuid)
                    return myid => gpu_id
                end
            end
            if remove_oversubscribed_workers
                rmprocs(myid)
                return myid => nothing
            else
                error("Can't assign a unique GPU to every worker, process $myid has no free GPUs left.")
            end
        end)
        @everywhere workers() device!($assignments[myid()])
    end
    print_info && proc_info()
    nothing
end


"""
    proc_info()

Prints info about available processes and the GPUs they are assigned.
"""
function proc_info()
    @eval Main using Distributed
    lines = @eval Main map(procs()) do id
        @fetchfrom id begin
            info = ["myid = $id"]
            !isnothing($_mpi_rank()) && push!(info, "mpi-rank = $($_mpi_rank())")
            push!(info, "host = $(gethostname())")
            @isdefined(CUDA) && push!(info, "device = $(sprint(io->show(io, MIME("text/plain"), CUDA.device()))) $(split(string(CUDA.uuid(CUDA.device())),'-')[1]))")
            " ("*join(info, ", ")*")"
        end
    end
    @info join(["Processes ($(nprocs())):"; lines], "\n")
end

_mpi_rank() = @eval Main @isdefined(MPI) && MPI.Initialized() ? MPI.Comm_rank(MPI.COMM_WORLD) : nothing

end
