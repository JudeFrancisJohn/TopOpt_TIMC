# Quick test to exercise export_vtk for one iteration
# This script does a minimal include of the project and calls export_vtk to
# produce one .vtu file inside the current run folder.

# Use project environment
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))

include("../src/stochastic_modified_v2.jl")

# At this point, `save_path`, `u`, `dh`, `grid`, `cv_post`, `mp`, `ip` should be in scope
# from including the main file. We will call export_vtk once using the current state.

println("save_path = ", save_path)

# Use a short test id
test_id = "quick_test"

# Ensure the save path exists
mkpath(save_path)

# Call export
export_vtk(u, dh, grid, cv_post, mp, ip, save_path, test_id; density = x)

full_vtu = joinpath(save_path, string(test_id) * ".vtu")
println("Wrote: ", full_vtu)

# List files in that folder
println("Files in run folder:")
for f in readdir(save_path)
    println(" - ", f)
end
