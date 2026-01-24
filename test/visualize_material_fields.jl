"""
visualize_material_fields.jl

Visualize the reconstructed spatial material fields (μ_l, μ_t, α, β) as heatmaps
at element level, using the same reconstruction procedure as
`test/reconstruct_best_vtu.jl`.

Usage (from repo root):
    julia --project=. test/visualize_material_fields.jl <RUN_DIR>

This script will:
- load eigenmodes.jld2 and optimization_history.jld2 for the given run directory
- reconstruct per-node material fields via sample_KL_field
- average to element-centroid values
- generate and save heatmap PNGs for each varying property

Outputs will be saved inside <RUN_DIR> as:
    material_field_mu_l.png
    material_field_mu_t.png
    material_field_alpha.png
    material_field_beta.png

Notes:
- We include the same proxy driver to ensure grid/mesh globals (coords_elem, nelx, nely) match.
- Heatmaps are arranged according to (nely, nelx) element layout with origin at lower-left.
"""

using JLD2
using Random
using Printf
using Statistics
using Plots

println("Loading TopOpt driver for visualization...")
include("../src/COPY_stochastic_modified_v2 copy 2_proxy.jl")
include("../input/params_MCMC.jl")
include("../utils/adversarial_utils.jl")

"""
    load_kl_and_coeffs(run_dir::AbstractString)

Shared helper to load KL modes and best coefficients (same as reconstruct script).
"""
function load_kl_and_coeffs(run_dir::AbstractString)
    eigen_file = joinpath(run_dir, "eigenmodes.jld2")
    history_file = joinpath(run_dir, "optimization_history.jld2")

    @assert isfile(eigen_file) "Missing eigenmodes.jld2 in $(run_dir)"
    @assert isfile(history_file) "Missing optimization_history.jld2 in $(run_dir)"

    kl_modes = nothing
    n_modes = nothing
    properties = nothing
    @info "Loading KL eigenmodes from: $eigen_file"
    jldopen(eigen_file, "r") do f
        kl_modes = read(f, "kl_modes")
        n_modes = read(f, "n_modes")
    end

    @info "Loading best coefficients + property order from: $history_file"
    best_coeffs = nothing
    jldopen(history_file, "r") do f
        best_coeffs = read(f, "best_coeffs")
        properties = Tuple(read(f, "properties"))
    end

    return kl_modes, n_modes, properties, best_coeffs
end

"""
    reconstruct_material_field(run_dir::AbstractString)

Reconstruct MaterialField using saved KL modes and best coefficients.
Returns the MaterialField and the properties tuple.
"""
function reconstruct_material_field(run_dir::AbstractString)
    SEED = 42
    Random.seed!(SEED)

    kl_modes_dict, n_modes, properties, best_coeffs_mat = load_kl_and_coeffs(run_dir)
    coeffs_dict = matrix_to_coeffs_dict(best_coeffs_mat, properties, n_modes)

    result_fields = Dict{Symbol, Any}()
    for prop_sym in properties
        field = sample_KL_field(kl_modes_dict[prop_sym], coeffs_dict[prop_sym]; eltype_out=Float32)
        result_fields[prop_sym] = field
    end
    # Constant properties
    result_fields[:λ] = fill(Float32(mp.λ), size(coords_elem,1), size(coords_elem,2))
    result_fields[:angle] = fill(Float32(mp.angle), size(coords_elem,1), size(coords_elem,2))

    mf = build_material_field(result_fields; use_centroids=false, eltype_out=Float32)
    return mf, properties
end

"""
    element_average(field_mat::AbstractMatrix)

Average per-element values over local nodes to obtain one scalar per element.
Input shape: (nelem, nloc)
Returns: Vector{Float64} of length nelem.
"""
function element_average(field_mat::AbstractMatrix)
    nelem, nloc = size(field_mat)
    avg = zeros(Float64, nelem)
    @inbounds for e in 1:nelem
        s = 0.0
        for i in 1:nloc
            s += field_mat[e, i]
        end
        avg[e] = s / nloc
    end
    return avg
end

"""
    field_vector_to_grid(avg_vec::AbstractVector, nelx::Int, nely::Int)

Reshape a length (nelx*nely) vector into a (nely, nelx) matrix for heatmap.
Assumes lexicographic ordering consistent with project conventions.
"""
function field_vector_to_grid(avg_vec::AbstractVector, nelx::Int, nely::Int)
    @assert length(avg_vec) == nelx * nely
    # Reshape row-major then flip vertically for bottom-left origin visualization
    mat = reshape(avg_vec, (nelx, nely))'  # now (nely, nelx)
    mat_flipped = reverse(mat, dims=1)
    return mat_flipped
end

"""
    plot_and_save_heatmap(run_dir::AbstractString, name::Symbol, grid_data; cmap=:viridis)

Create and save a heatmap PNG in the run directory.
"""
function plot_and_save_heatmap(run_dir::AbstractString, name::Symbol, grid_data; cmap=:viridis)
    plt = heatmap(grid_data; color=cmap, aspect_ratio=:equal, title=string(name), xlabel="x (elements)", ylabel="y (elements)")
    outpath = joinpath(run_dir, @sprintf("material_field_%s.png", String(name)))
    savefig(plt, outpath)
    println("Saved heatmap: $(outpath)")
end

"""
Main entry: reconstruct material fields and save heatmaps.
"""
function main()
    if length(ARGS) < 1
        println("Usage: julia --project=. test/visualize_material_fields.jl <RUN_DIR>")
        return
    end
    run_dir = ARGS[1]
    @info "Visualizing material fields" run_dir

    mf, properties = reconstruct_material_field(run_dir)

    # Determine grid dimensions
    nelem = size(coords_elem, 1)
    nloc = size(coords_elem, 2)
    @assert nelem == nelx * nely "Element count mismatch with (nelx*nely)"

    # Select properties that vary (from reconstruction set)
    props_to_plot = [p for p in properties]

    # Average to per-element scalars and plot
    for prop_sym in props_to_plot
        field_mat = getfield(mf, prop_sym)
        avg_vec = element_average(field_mat)
        grid_data = field_vector_to_grid(avg_vec, nelx, nely)
        plot_and_save_heatmap(run_dir, prop_sym, grid_data)
    end

    println("Visualization complete.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
