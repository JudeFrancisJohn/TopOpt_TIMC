"""
reconstruct_best_vtu.jl

Recreate the best-iteration VTU from a finished adversarial run using the
saved KL eigenmodes and the recorded best coefficients. This does NOT run the
outer optimizer again; it rebuilds material fields and runs a single TopOpt
solve like `evaluate_objective` did in `test/proxy.jl`.

Usage (from repo root):
    julia --project=. test/reconstruct_best_vtu.jl <RUN_DIR>

Where <RUN_DIR> is something like:
    output/adversarial_YYYYMMDD_HHMMSS

This script will:
- load eigenmodes.jld2 and optimization_history.jld2 (for best_coeffs, properties)
- reconstruct material fields via sample_KL_field
- rebuild KE_store
- run TopOpt once
- write a VTU named best_iteration_####_reconstructed.vtu (or same tag if desired)

Note: we include the same proxy driver used by test/proxy.jl to ensure the
FE/TopOpt environment is identical.
"""

using JLD2
using Random
using Dates
using Printf

# Load project modules in the same way as proxy.jl
println("Loading TopOpt driver for reconstruction...")
include("../src/COPY_stochastic_modified_v2 copy 2_proxy.jl")
include("../input/params_MCMC.jl")
include("../utils/adversarial_utils.jl")

# ---------- Helpers ----------

"""
    load_kl_and_coeffs(run_dir::AbstractString)

Load KL eigenmodes and the best coefficients (matrix form) along with properties
and n_modes from the given adversarial run directory.
"""
function load_kl_and_coeffs(run_dir::AbstractString)
    eigen_file = joinpath(run_dir, "eigenmodes.jld2")
    history_file = joinpath(run_dir, "optimization_history.jld2")

    @assert isfile(eigen_file) "Missing eigenmodes.jld2 in $(run_dir)"
    @assert isfile(history_file) "Missing optimization_history.jld2 in $(run_dir)"

    # Load eigenmodes
    kl_modes = nothing
    n_modes = nothing
    σs = nothing
    @info "Loading KL eigenmodes from: $eigen_file"
    jldopen(eigen_file, "r") do f
        kl_modes = read(f, "kl_modes")
        n_modes = read(f, "n_modes")
        σs = haskey(f, "σs") ? read(f, "σs") : nothing
    end

    # Load best coefficients + properties ordering
    best_coeffs = nothing
    properties = nothing
    @info "Loading best coefficients from: $history_file"
    jldopen(history_file, "r") do f
        best_coeffs = read(f, "best_coeffs")
        properties = Tuple(read(f, "properties"))
    end

    return kl_modes, n_modes, properties, best_coeffs
end

"""
    tag_from_metadata(run_dir::AbstractString; fallback::String="best_reconstructed")

Read iteration from best_metadata.txt if present to build a tag like
"best_iteration_0038". Otherwise return the fallback tag.
"""
function tag_from_metadata(run_dir::AbstractString; fallback::String="best_reconstructed")
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

"""
    reconstruct_best_vtu(run_dir::AbstractString; overwrite::Bool=false)

Rebuild the material fields and run a single TopOpt using the best coefficients.
Writes a VTU file inside `run_dir` with a tag derived from metadata (or fallback).
Returns (X, c, vtu_path).
"""
function reconstruct_best_vtu(run_dir::AbstractString; overwrite::Bool=false)
    # Match proxy.jl seed for determinism if needed
    SEED = 42
    Random.seed!(SEED)

    # Load KL and best coefficients
    kl_modes_dict, n_modes, properties, best_coeffs_mat = load_kl_and_coeffs(run_dir)
    
    println("\n[RECONSTRUCTION DEBUG]")
    println("  Properties from saved data: ", properties)
    println("  KL modes available for: ", keys(kl_modes_dict))
    println("  Best coeffs matrix size: ", size(best_coeffs_mat))

    # Convert matrix -> dict aligned with properties order
    coeffs_dict = matrix_to_coeffs_dict(best_coeffs_mat, properties, n_modes)
    
    println("  Coeffs dict keys: ", keys(coeffs_dict))
    for (k, v) in coeffs_dict
        println("    $k: ", length(v), " coefficients")
    end

    # Generate material fields for each optimized property (using Greek symbols as stored)
    result_fields = Dict{Symbol, Any}()
    for prop_sym in properties
        field = sample_KL_field(kl_modes_dict[prop_sym], coeffs_dict[prop_sym]; eltype_out=Float32)
        result_fields[prop_sym] = field
        println("  Generated field for $prop_sym: size ", size(field))
    end
    
    # Map Greek to Latin for build_material_field (it expects Latin names)
    # Also add constant properties for any not sampled via KL
    GREEK_TO_LATIN = Dict(:α => :alpha, :β => :beta)
    nelem, nloc = size(coords_elem)
    
    # Create Latin-named dict for build_material_field
    latin_fields = Dict{Symbol, Any}()
    for (greek_sym, field) in result_fields
        latin_sym = get(GREEK_TO_LATIN, greek_sym, greek_sym)
        latin_fields[latin_sym] = field
        println("  Mapped $greek_sym -> $latin_sym")
    end
    
    # Fill in any missing required properties with constants
    for prop_sym in (:μ_l, :μ_t, :alpha, :beta, :λ, :angle)
        if !haskey(latin_fields, prop_sym)
            val = getfield(mp, prop_sym)
            latin_fields[prop_sym] = fill(Float32(val), nelem, nloc)
            println("  Added constant field for $prop_sym = $val")
        end
    end

    # Build MaterialField
    mf = build_material_field(latin_fields; use_centroids=false, eltype_out=Float32)

    # Rebuild KE store for FE
    build_KEStore!(dh, mf, nnodes_loc, avg_mp_store)

    # Reset displacement vector and run TopOpt once (like evaluate_objective)
    global u = zeros(ndofs(dh))
    X, c = topopt_run(1)

    # Determine output tag and path
    tag = tag_from_metadata(run_dir; fallback="best_reconstructed")
    save_dir = run_dir
    mkpath(save_dir)

    # Compose full path; optionally avoid overwriting existing
    base_path = joinpath(save_dir, tag * ".vtu")
    vtu_path = base_path
    if isfile(base_path) && !overwrite
        vtu_path = joinpath(save_dir, tag * "_reconstructed.vtu")
    end

    # Export VTU with material field
    export_vtk(u, dh, grid, cv_post, mp, ip, save_dir, splitext(basename(vtu_path))[1]; density=X, material_field=mf)
    println("Wrote: $(vtu_path) (with material field parameters)")

    return X, c, vtu_path
end

# If executed directly, use ARGS[1] as run_dir
if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 1
        println("Usage: julia --project=. test/reconstruct_best_vtu.jl <RUN_DIR>")
        exit(1)
    end
    run_dir = ARGS[1]
    @info "Reconstructing best VTU from run directory" run_dir
    reconstruct_best_vtu(run_dir)
end
