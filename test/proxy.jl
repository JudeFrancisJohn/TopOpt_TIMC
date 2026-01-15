"""
proxy.jl

Main driver for adversarial topology optimization.

This script searches for KL expansion coefficients that maximize intermediate
density regions in SIMP topology optimization. It uses a modular design with
pluggable optimization strategies (CMA-ES, Simulated Annealing, etc.).

Usage:
    julia --project=. test/proxy.jl
"""

using Random
using Dates
using JLD2

# ============================================================================
# LOAD DEPENDENCIES
# ============================================================================

# Load TopOpt driver and utilities
include("../src/COPY_stochastic_modified_v2 copy 2_proxy.jl")

# Load configuration parameters
include("../input/params_adversarial.jl")
include("../input/params_MCMC.jl")

# Load adversarial optimization modules
include("../utils/abstract_optimizer.jl")
include("../utils/adversarial_objective.jl")
include("../utils/adversarial_logging.jl")

# Load concrete optimizer implementations
include("../utils/cmaes_optimizer.jl")
# include("../utils/simulated_annealing_optimizer.jl")  # To be implemented

# ============================================================================
# SETUP
# ============================================================================

# Create output directory
const OUTPUT_ROOT = normpath(joinpath(@__DIR__, "..", "output"))
dt_str = Dates.format(Dates.now(), "yyyymmdd_HH")
const SAVE_PATH = joinpath(OUTPUT_ROOT, "adversarial_$(dt_str)")
mkpath(SAVE_PATH)

println("="^80)
println("ADVERSARIAL TOPOLOGY OPTIMIZATION")
println("="^80)
println("Output directory: $(SAVE_PATH)")
println("Optimizer type: $(OPTIMIZER_TYPE)")

# Set random seed
Random.seed!(ADVERSARIAL_SEED)
println("Random seed: $(ADVERSARIAL_SEED)")

# Get mesh info
const N_ELEM, N_LOC = size(coords_elem)
println("\nMesh info:")
println("  Elements: $(N_ELEM)")
println("  Nodes per element: $(N_LOC)")

# Define properties to optimize
const PROPERTIES = VARIABLE_PROPERTIES
const N_PROPS = length(PROPERTIES)

# ============================================================================
# PRE-COMPUTE OR LOAD KL EIGENMODES (WITH CACHING)
# ============================================================================

println("\n" * "="^80)
println("KL EIGENMODES SETUP")
println("="^80)

# Create cache directory and define cache file path
const CACHE_DIR = joinpath(OUTPUT_ROOT, "eigenmode_cache")
mkpath(CACHE_DIR)

# Get mesh info for cache key
const N_ELEM, N_LOC = size(coords_elem)

# Create cache key from parameters that affect eigenmodes
cache_key = "$(join(PROPERTIES, "_"))_Nmodes$(N_MODES_PER_PROP)_mesh$(N_ELEM)x$(N_LOC)"
cache_file = joinpath(CACHE_DIR, "$(cache_key).jld2")

kl_modes_dict = Dict{Symbol, Any}()
n_modes = Dict{Symbol, Int}()
loaded_from = nothing

# Strategy 1: Try dedicated cache directory first
if isfile(cache_file)
    println("✓ Found cached eigenmodes in cache directory")
    println("  File: $cache_file")
    println("  Loading from cache (skipping eigenvalue computation)...")
    
    try
        cached_data = JLD2.load(cache_file)
        kl_modes_dict = cached_data["kl_modes"]
        n_modes = cached_data["n_modes"]
        
        # Verify properties match
        if Set(keys(kl_modes_dict)) == Set(PROPERTIES)
            println("  ✓ Properties match: $(join(PROPERTIES, ", "))")
            println("  ✓ Using $(length(kl_modes_dict)) pre-computed eigenmode sets")
            println("  ✓ Total parameters: $(sum(values(n_modes)))")
            loaded_from = "cache"
        else
            @warn "Cached properties don't match current VARIABLE_PROPERTIES" cached=keys(kl_modes_dict) current=PROPERTIES
            kl_modes_dict = Dict{Symbol, Any}()
            n_modes = Dict{Symbol, Int}()
        end
    catch e
        @warn "Failed to load from cache directory" error=e
        kl_modes_dict = Dict{Symbol, Any}()
        n_modes = Dict{Symbol, Int}()
    end
end

# Strategy 2: If cache failed, look for recent eigenmodes.jld2 in previous runs
if isempty(kl_modes_dict)
    println("\n  Searching for eigenmodes in previous adversarial runs...")
    
    # Find all eigenmodes.jld2 files in adversarial_* directories
    eigenmode_files = String[]
    for dir in readdir(OUTPUT_ROOT; join=true)
        if isdir(dir) && occursin("adversarial_", basename(dir))
            eigenmode_path = joinpath(dir, "eigenmodes.jld2")
            if isfile(eigenmode_path)
                push!(eigenmode_files, eigenmode_path)
            end
        end
    end
    
    # Sort by modification time (most recent first)
    if !isempty(eigenmode_files)
        sort!(eigenmode_files; by=f -> stat(f).mtime, rev=true)
        
        println("  Found $(length(eigenmode_files)) existing eigenmode file(s)")
        
        # Try to load the most recent one
        for eigenmode_path in eigenmode_files
            println("  Trying: $eigenmode_path")
            
            try
                cached_data = JLD2.load(eigenmode_path)
                temp_kl_modes = cached_data["kl_modes"]
                temp_n_modes = cached_data["n_modes"]
                
                # Check if properties match
                if Set(keys(temp_kl_modes)) == Set(PROPERTIES)
                    kl_modes_dict = temp_kl_modes
                    n_modes = temp_n_modes
                    println("  ✓ Properties match! Using this file.")
                    println("  ✓ Loaded $(length(kl_modes_dict)) eigenmode sets")
                    println("  ✓ Total parameters: $(sum(values(n_modes)))")
                    loaded_from = "previous_run"
                    break
                else
                    println("  ✗ Properties don't match (has $(join(keys(temp_kl_modes), ", ")) vs need $(join(PROPERTIES, ", ")))")
                end
            catch e
                println("  ✗ Failed to load: $e")
            end
        end
    else
        println("  No previous eigenmode files found")
    end
end

if isempty(kl_modes_dict)
    println("\n  → No usable cached eigenmodes found - will compute from scratch")
end

# Compute eigenmodes if cache was invalid or missing
if isempty(kl_modes_dict)
    println("\n" * "="^80)
    println("COMPUTING KL EIGENMODES")
    println("="^80)
    
    for prop_sym in PROPERTIES
        sigma = get(σs_ADVERSARIAL, prop_sym, 0.5)
        
        println("\nComputing eigenmodes for $prop_sym...")
        println("  σ = $sigma, N_modes = $N_MODES_PER_PROP")
        
        mode_type = (prop_sym in (:μ_l, :μ_t, :λ)) ? :lognormal : :additive
        
        kl_modes = compute_KL_eigenmodes(
            mp, coords_elem, prop_sym, sigma;
            Lc=Lc_ADVERSARIAL,
            N_modes=N_MODES_PER_PROP,
            use_centroids=false,
            make_sparse=false,
            mode=mode_type
        )
        
        kl_modes_dict[prop_sym] = kl_modes
        n_modes[prop_sym] = length(kl_modes.eigenvalues)
        
        println("  ✓ Computed $(n_modes[prop_sym]) eigenmodes (mode: $mode_type)")
        println("  ✓ Eigenvalue range: [$(minimum(kl_modes.eigenvalues)), $(maximum(kl_modes.eigenvalues))]")
    end
    
    # Save to cache for future runs
    println("\n💾 Saving eigenmodes to cache for future runs...")
    println("  Cache file: $cache_file")
    jldsave(cache_file; kl_modes=kl_modes_dict, n_modes=n_modes, σs=σs_ADVERSARIAL)
    println("  ✓ Cache saved successfully")
    loaded_from = "newly_computed"
    
    println("\n" * "="^80)
    println("EIGENMODE COMPUTATION COMPLETE")
    println("="^80)
    println("Total parameters to optimize: $(sum(values(n_modes)))")
end

# Summary of where eigenmodes came from
println("\n" * "="^80)
if loaded_from == "cache"
    println("📂 EIGENMODES LOADED FROM CACHE (instant)")
elseif loaded_from == "previous_run"
    println("📂 EIGENMODES REUSED FROM PREVIOUS RUN (instant)")
elseif loaded_from == "newly_computed"
    println("🔧 EIGENMODES COMPUTED FROM SCRATCH")
else
    println("⚠️  EIGENMODE SOURCE UNKNOWN")
end
println("="^80)

# Also save eigenmodes in run directory for reproducibility
eigenmode_file = joinpath(SAVE_PATH, "eigenmodes.jld2")
jldsave(eigenmode_file; kl_modes=kl_modes_dict, n_modes=n_modes, σs=σs_ADVERSARIAL)
println("\nSaved eigenmodes copy to run directory: $eigenmode_file")

# ============================================================================
# CREATE OPTIMIZER CONFIGURATION
# ============================================================================

opt_config = OptimizerConfig(
    n_modes, PROPERTIES;
    coeff_bounds=(COEFF_LOWER_BOUND, COEFF_UPPER_BOUND),
    w_frac=W_FRAC_ADVERSARIAL,
    w_severity=W_SEVERITY_ADVERSARIAL,
    w_gray=W_GRAY_ADVERSARIAL,
    w_stability=W_STABILITY_ADVERSARIAL,
    save_path=SAVE_PATH,
    save_every=SAVE_EVERY_ADVERSARIAL,
    seed=ADVERSARIAL_SEED
)

# ============================================================================
# SELECT AND CONFIGURE OPTIMIZER
# ============================================================================

optimizer = if OPTIMIZER_TYPE == :cmaes
    CMAESOptimizer(
        opt_config;
        max_iterations=MAX_ITERATIONS_ADVERSARIAL,
        population_size=POPULATION_SIZE_ADVERSARIAL,
        initial_sigma=INITIAL_SIGMA_ADVERSARIAL
    )
elseif OPTIMIZER_TYPE == :simulated_annealing
    # SimulatedAnnealingOptimizer(opt_config; ...)  # To be implemented
    error("Simulated Annealing optimizer not yet implemented")
else
    error("Unknown optimizer type: $(OPTIMIZER_TYPE)")
end

println("\nOptimizer configured:")
println("  Type: $(OPTIMIZER_TYPE)")
println("  Max iterations: $(MAX_ITERATIONS_ADVERSARIAL)")

# ============================================================================
# CREATE OBJECTIVE FUNCTION
# ============================================================================

# Create export wrapper for VTU saving
function export_vtk_wrapper(save_dir::String, filename_prefix::String, X::Vector{Float64})
    export_vtk(u, dh, grid, cv_post, mp, ip, save_dir, filename_prefix; density=X)
end

# Create objective function closure
objective_fn = create_objective_function(
    kl_modes_dict, PROPERTIES, n_modes,
    mp, coords_elem, dh, mf, avg_mp_store,
    topopt_run, build_KEStore!;
    validation_tolerance=VALIDATION_TOLERANCE_FACTOR,
    output_root=OUTPUT_ROOT
)

# ============================================================================
# RUN OPTIMIZATION
# ============================================================================

println("\n" * "="^80)
println("STARTING OPTIMIZATION")
println("="^80)

# Initialize coefficients
initial_coeffs = initialize_coefficients(
    n_modes, PROPERTIES;
    seed=ADVERSARIAL_SEED,
    initial_sigma=INITIAL_SIGMA_ADVERSARIAL
)

# Run optimization (polymorphic call - works for any optimizer type)
best_coeffs, best_badness, opt_result = optimize!(
    optimizer, 
    objective_fn, 
    initial_coeffs,
    export_vtk_wrapper
)

# ============================================================================
# FINALIZE
# ============================================================================

println("\n" * "="^80)
println("OPTIMIZATION COMPLETE!")
println("="^80)
println("\nBest badness: $(round(best_badness, digits=6))")
println("\nResults saved to: $SAVE_PATH")
println("\nTo reconstruct material fields from best coefficients:")
println("  1. Load coefficients from: best_coefficients.txt")
println("  2. Load eigenmodes from: eigenmodes.jld2")
println("  3. Use `sample_KL_field()` with pre-computed eigenmodes")
println("  4. Build MaterialField and run TopOpt")

println("\n✅ Done!")
