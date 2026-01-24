using JLD2
using DelimitedFiles
using Statistics
using Printf
using Random

# ============================================================================
# CONFIGURATION
# ============================================================================

# Path to the results we want to reproduce
const RESULTS_DIR = joinpath(@__DIR__, "..", "output", "adversarial_20251203_163054")
const COEFFS_FILE = joinpath(RESULTS_DIR, "best_coefficients.txt")
const EIGENMODES_FILE = joinpath(RESULTS_DIR, "eigenmodes.jld2")

println("="^80)
println("REPLOTTING ADVERSARIAL RESULT")
println("="^80)
println("Results directory: $RESULTS_DIR")

# ============================================================================
# LOAD DRIVER
# ============================================================================

# Set save_path BEFORE including the driver (it uses it)
# We want to save in the SAME directory as the results
save_path = RESULTS_DIR

println("\nLoading TopOpt driver...")
include("../src/replot_driver.jl")

# Define aliases for consistency with proxy.jl
const N_ELEM = nelem
const N_LOC = nnodes_per_cell

# ============================================================================
# RECONSTRUCT EIGENMODES
# ============================================================================

println("\n" * "="^80)
println("LOADING / COMPUTING EIGENMODES")
println("="^80)

kl_modes_dict = Dict{Symbol, Any}()
n_modes = Dict{Symbol, Int}()

# Parameters from proxy.jl (MUST MATCH THE RUN!)
const PROPERTIES = (:μ_l, :μ_t, :α, :β)
const σs = Dict(
    :μ_l => 1.5,
    :μ_t => 1.5,
    :α   => 1.5,
    :β   => 1.5
)
const N_MODES_PER_PROP = 15
const Lc_val = 2.0

if isfile(EIGENMODES_FILE)
    println("Loading eigenmodes from file: $EIGENMODES_FILE")
    data = load(EIGENMODES_FILE)
    global kl_modes_dict = data["kl_modes"]
    global n_modes = data["n_modes"]
else
    println("Eigenmodes file not found. Recomputing with original parameters...")
    println("  Lc = $Lc_val")
    println("  N_modes = $N_MODES_PER_PROP")
    println("  σs = $σs")
    
    for prop_sym in PROPERTIES
        sigma = get(σs, prop_sym, 0.5)
        println("  Computing eigenmodes for $prop_sym...")
        
        mode_type = (prop_sym == :μ_l || prop_sym == :μ_t) ? :lognormal : :additive
        
        kl_modes = compute_KL_eigenmodes(
            mp, coords_elem, prop_sym, sigma;
            Lc=Lc_val,
            N_modes=N_MODES_PER_PROP, 
            use_centroids=false,
            make_sparse=false,
            mode=mode_type
        )
        
        kl_modes_dict[prop_sym] = kl_modes
        n_modes[prop_sym] = length(kl_modes.eigenvalues)
    end
end

# ============================================================================
# LOAD COEFFICIENTS
# ============================================================================

println("\n" * "="^80)
println("LOADING COEFFICIENTS")
println("="^80)

function parse_coefficients(filepath)
    coeffs_dict = Dict{Symbol, Vector{Float64}}()
    
    # Initialize vectors
    for p in PROPERTIES
        coeffs_dict[p] = zeros(N_MODES_PER_PROP)
    end
    
    lines = readlines(filepath)
    current_prop = nothing
    
    for line in lines
        line = strip(line)
        if isempty(line) || startswith(line, "#")
            continue
        end
        
        # Check for property header like "## μ_l (15 modes)"
        if startswith(line, "##")
            if contains(line, "μ_l") current_prop = :μ_l
            elseif contains(line, "μ_t") current_prop = :μ_t
            elseif contains(line, "α") || contains(line, "alpha") current_prop = :α
            elseif contains(line, "β") || contains(line, "beta") current_prop = :β
            end
            continue
        end
        
        # Parse coefficient line: "1       μ_l  1.9874324317e+00"
        if current_prop !== nothing
            parts = split(line)
            if length(parts) >= 3
                try
                    idx = parse(Int, parts[1])
                    val = parse(Float64, parts[3])
                    if idx <= N_MODES_PER_PROP
                        coeffs_dict[current_prop][idx] = val
                    end
                catch
                    # Ignore parsing errors (headers etc)
                end
            end
        end
    end
    return coeffs_dict
end

coeffs_dict = parse_coefficients(COEFFS_FILE)
println("Loaded coefficients:")
for (k, v) in coeffs_dict
    println("  $k: $(length(v)) values, mean=$(mean(v))")
end

# ============================================================================
# RECONSTRUCT MATERIAL FIELD
# ============================================================================

println("\n" * "="^80)
println("RECONSTRUCTING MATERIAL FIELD")
println("="^80)

result_fields = Dict{Symbol, Any}()

for prop_sym in PROPERTIES
    coeffs_for_prop = coeffs_dict[prop_sym]
    field = sample_KL_field(
        kl_modes_dict[prop_sym], 
        coeffs_for_prop; 
        eltype_out=Float32
    )
    result_fields[prop_sym] = field
end

# Add constant properties
result_fields[:λ] = fill(Float32(mp.λ), N_ELEM, N_LOC)
result_fields[:angle] = fill(Float32(mp.angle), N_ELEM, N_LOC)

mf = build_material_field(result_fields; use_centroids=false, eltype_out=Float32)
println("Material field built successfully.")

# ============================================================================
# RUN TOPOPT
# ============================================================================

println("\n" * "="^80)
println("RUNNING TOPOPT RECONSTRUCTION")
println("="^80)

# Build stiffness matrices
println("Building stiffness matrices...")
build_KEStore!(dh, mf, nnodes_loc, avg_mp_store)

# Initialize displacement
global u = zeros(ndofs(dh))

# Run TopOpt
println("Starting optimization...")
X, c = topopt_run(1)

println("\n" * "="^80)
println("RECONSTRUCTION COMPLETE")
println("="^80)
println("Output saved to: $save_path")
