"""
export_material_fields_vtu.jl

Reconstruct the adversarial best material fields and export them as cell-data
fields in a VTU file so they can be visualized as 3D heatmaps in ParaView.

Usage (from repo root):
    julia --project=. test/export_material_fields_vtu.jl <RUN_DIR> [--tag TAG]

- <RUN_DIR> should be the adversarial output folder (e.g. output/adversarial_YYYYMMDD_HHMMSS)
- Optional --tag lets you choose the filename base; by default this script will
  try to read iteration from best_metadata.txt and use best_iteration_####, else
  it falls back to material_fields.

This script does NOT modify project utilities. It includes the proxy driver to
access `grid`, `dh`, mesh sizes, and uses WriteVTK directly to create a VTU
with per-element fields: μ_l, μ_t, α, β, λ, angle, and (optionally) density X
if available from a TopOpt run.
"""

using JLD2
using Random
using Statistics
using Printf
using Dates
using WriteVTK

println("Loading TopOpt driver for field export...")
include("../src/COPY_stochastic_modified_v2 copy 2_proxy.jl")
include("../input/params_MCMC.jl")
include("../utils/adversarial_utils.jl")
include("reconstruct_best_vtu.jl")  # reuse helpers and optionally run TopOpt

# ------------------ helpers ------------------

function tag_from_metadata(run_dir::AbstractString; fallback::String="material_fields")
    meta = joinpath(run_dir, "best_metadata.txt")
    if isfile(meta)
        iter_val = nothing
        for ln in eachline(meta)
            if startswith(ln, "iteration:")
                parts = split(ln, ":")
                if length(parts) >= 2
                    iter_val = tryparse(Int, strip(parts[2]))
                end
            end
        end
        if iter_val !== nothing
            return @sprintf("best_iteration_%04d", iter_val)
        end
    end
    return fallback
end

function load_kl_and_coeffs(run_dir::AbstractString)
    eigen_file = joinpath(run_dir, "eigenmodes.jld2")
    history_file = joinpath(run_dir, "optimization_history.jld2")
    @assert isfile(eigen_file) "Missing eigenmodes.jld2 in $(run_dir)"
    @assert isfile(history_file) "Missing optimization_history.jld2 in $(run_dir)"
    kl_modes = nothing
    n_modes = nothing
    properties = nothing
    jldopen(eigen_file, "r") do f
        kl_modes = read(f, "kl_modes")
        n_modes = read(f, "n_modes")
    end
    best_coeffs = nothing
    jldopen(history_file, "r") do f
        best_coeffs = read(f, "best_coeffs")
        properties = Tuple(read(f, "properties"))
    end
    return kl_modes, n_modes, properties, best_coeffs
end

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
    # constants
    nelem, nloc = size(coords_elem)
    
    # Add constant properties for any not sampled via KL
    for prop_sym in (:μ_l, :μ_t, :α, :β, :λ, :angle)
        if !haskey(result_fields, prop_sym)
            val = prop_sym == :λ ? mp.λ : (prop_sym == :angle ? mp.angle : getfield(mp, Base.Meta.parse(string(prop_sym))))
            result_fields[prop_sym] = fill(Float32(val), nelem, nloc)
        end
    end
    mf = build_material_field(result_fields; use_centroids=false, eltype_out=Float32)
    return mf, properties
end

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

# ------------------ main export ------------------

function main()
    if length(ARGS) < 1
        println("Usage: julia --project=. test/export_material_fields_vtu.jl <RUN_DIR> [--tag TAG]")
        return
    end
    run_dir = ARGS[1]
    arg_tag = nothing
    if length(ARGS) >= 3 && ARGS[2] == "--tag"
        arg_tag = ARGS[3]
    end

    # reconstruct material field
    mf, properties = reconstruct_material_field(run_dir)

    # optional: also run a single TopOpt to get X for export (density)
    # this reuses reconstruct_best_vtu without overwriting original file
    X = nothing
    try
        X, _, _ = reconstruct_best_vtu(run_dir; overwrite=false)
    catch e
        @warn "Could not reconstruct density field X; proceeding with material fields only" exception=e
    end

    # build cell-data arrays (per element)
    mu_l = element_average(mf.μ_l)
    mu_t = element_average(mf.μ_t)
    alpha = element_average(mf.α)
    beta = element_average(mf.β)
    lam = element_average(mf.λ)
    ang = element_average(mf.angle)

    # Create a new VTU with cell data
    tag = isnothing(arg_tag) ? tag_from_metadata(run_dir; fallback="material_fields") : arg_tag
    base = joinpath(run_dir, tag)

    vtkfile = vtk_grid(base, grid)
    vtk_cell_data(vtkfile, mu_l, "mu_l")
    vtk_cell_data(vtkfile, mu_t, "mu_t")
    vtk_cell_data(vtkfile, alpha, "alpha")
    vtk_cell_data(vtkfile, beta, "beta")
    vtk_cell_data(vtkfile, lam, "lambda")
    vtk_cell_data(vtkfile, ang, "angle")
    if X !== nothing
        vtk_cell_data(vtkfile, X, "density")
    end
    # also include displacement if available
    try
        vtk_point_data(vtkfile, u, "displacement")
    catch
        # ignore if u not set
    end

    outpath = vtk_save(vtkfile)
    println("Wrote material-field VTU: $(outpath)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
