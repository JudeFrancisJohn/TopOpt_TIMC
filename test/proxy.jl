"""
proxy.jl

Adversarial optimization for finding KL expansion coefficients that maximize
intermediate density regions in SIMP topology optimization.

This script:
1. Pre-computes KL eigenmodes ONCE (deterministic)
2. Uses CMA-ES to search for coefficients that create confusing material heterogeneities
3. Evaluates each candidate by running full TopOpt and measuring intermediate density metrics
4. Saves best coefficients and material parameter fields
"""

using Random
using Statistics
using Dates
using Printf
using BlackBoxOptim  # CMA-ES implementation
using JLD2

# Load project modules
println("Loading TopOpt driver...")
include("../src/COPY_stochastic_modified_v2 copy 2_proxy.jl")  # Use proxy version, not MC version!
include("../input/params_MCMC.jl")
include("../utils/adversarial_utils.jl")
include("../utils/adversarial_optimizer.jl")

# ----------------------------------------------------------------------------
# Best iteration tracking and VTU management
# ----------------------------------------------------------------------------

mutable struct BestVTUTracker
    badness::Float64
    compliance::Float64
    iter::Int
    vtu_path::Union{String, Nothing}
end

function init_best_tracker()
    return BestVTUTracker(-Inf, Inf, 0, nothing)
end

function delete_previous_vtu!(tracker::BestVTUTracker)
    if tracker.vtu_path !== nothing && isfile(tracker.vtu_path)
        try
            rm(tracker.vtu_path; force=true)
            println("    🗑️  Deleted previous best VTU: $(tracker.vtu_path)")
        catch e
            @warn "Failed to delete previous best VTU" path=tracker.vtu_path error=e
        end
    end
end

function save_best_vtu!(tracker::BestVTUTracker, X::AbstractVector{<:Real}, iter_id::Int; save_dir::AbstractString=SAVE_PATH)
    # Ensure directory exists
    mkpath(save_dir)
    # Compose filename
    fname = joinpath(save_dir, @sprintf("best_iteration_%04d.vtu", iter_id))
    # Export using project helper; density=X
    try
        export_vtk(u, dh, grid, cv_post, mp, ip, save_dir, @sprintf("best_iteration_%04d", iter_id); density=X)
        tracker.vtu_path = fname
        println("    💾 Saved best VTU: $(fname)")
    catch e
        @warn "Failed to export best VTU" iter=iter_id error=e
    end
end

function persist_best_metadata(save_dir::AbstractString, tracker::BestVTUTracker)
    meta_file = joinpath(save_dir, "best_metadata.txt")
    open(meta_file, "w") do io
        println(io, "Best iteration metadata")
        println(io, "timestamp: ", Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))
        println(io, @sprintf("badness: %.6f", tracker.badness))
        println(io, @sprintf("compliance: %.6f", tracker.compliance))
        println(io, "iteration: ", tracker.iter)
        println(io, "vtu_path: ", tracker.vtu_path === nothing ? "" : tracker.vtu_path)
    end
    println("    📝 Wrote metadata: $(meta_file)")
end

# ============================================================================
# CONFIGURATION
# ============================================================================

const OUTPUT_ROOT = normpath(joinpath(@__DIR__, "..", "output"))
dt_str = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
const SAVE_PATH = joinpath(OUTPUT_ROOT, "adversarial_$(dt_str)")
mkpath(SAVE_PATH)

println("=" * "="^80)
println("ADVERSARIAL COEFFICIENT OPTIMIZATION")
println("=" * "="^80)
println("Output directory: $(SAVE_PATH)")

# Set random seed for reproducibility
const SEED = 42
Random.seed!(SEED)
println("Random seed: $(SEED)")

# Define properties to optimize (from input/params_mat.jl via VARIABLE_PROPERTIES)
const PROPERTIES = VARIABLE_PROPERTIES
const N_PROPS = length(PROPERTIES)

# KL expansion parameters - INCREASED for stronger material heterogeneity
const σs = Dict(
    :μ_l => 1.5,
    :μ_t => 1.5,
    :α   => 1.5,
    :β   => 1.5,
    :λ   => 0.5,
    :angle => 1.0,
)

# Number of modes per property (REDUCED for efficiency)
# Start with fewer modes for faster optimization
const N_MODES_PER_PROP = 15  # Reduced from 80 to 15 for tractability

# Get mesh dimensions
const N_ELEM, N_LOC = size(coords_elem)
println("\nMesh info:")
println("  Elements: $(N_ELEM)")
println("  Nodes per element: $(N_LOC)")

# ============================================================================
# PRE-COMPUTE KL EIGENMODES (ONCE - DETERMINISTIC)
# ============================================================================

println("\n" * "="^80)
println("PRE-COMPUTING KL EIGENMODES")
println("="^80)

kl_modes_dict = Dict{Symbol, Any}()
n_modes = Dict{Symbol, Int}()

for prop_sym in PROPERTIES
    sigma = get(σs, prop_sym, 0.5)
    
    println("\nComputing eigenmodes for $prop_sym...")
    println("  σ = $sigma")
    println("  N_modes = $N_MODES_PER_PROP")
    
    # Use lognormal mode for positive-valued properties; additive for others
    mode_type = (prop_sym in (:μ_l, :μ_t, :λ)) ? :lognormal : :additive
    
    kl_modes = compute_KL_eigenmodes(
        mp, coords_elem, prop_sym, sigma;
        Lc=2.0,  # Increased from 0.01 to 2.0 (4x element size for meaningful spatial variation)
        N_modes=N_MODES_PER_PROP, 
        use_centroids=false,
        make_sparse=false,
        mode=mode_type  # Use lognormal for μ_l, μ_t to ensure positivity
    )
    
    kl_modes_dict[prop_sym] = kl_modes
    n_modes[prop_sym] = length(kl_modes.eigenvalues)
    
    println("  ✓ Computed $(n_modes[prop_sym]) eigenmodes (mode: $mode_type)")
    println("  ✓ Eigenvalue range: [$(minimum(kl_modes.eigenvalues)), $(maximum(kl_modes.eigenvalues))]")
end

println("\n" * "="^80)
println("EIGENMODE COMPUTATION COMPLETE")
println("="^80)
println("Total parameters to optimize: $(sum(values(n_modes)))")

# Save eigenmodes for later reconstruction
eigenmode_file = joinpath(SAVE_PATH, "eigenmodes.jld2")
println("\nSaving eigenmodes to: $eigenmode_file")
jldsave(eigenmode_file; kl_modes=kl_modes_dict, n_modes=n_modes, σs=σs)

# ============================================================================
# OBJECTIVE FUNCTION EVALUATION
# ============================================================================

"""
    evaluate_objective(coeffs_mat::Matrix{Float64})

Evaluate adversarial objective for given coefficient matrix.
Runs full topology optimization and computes badness metrics.

Returns: (badness, compliance, X, diagnostics)
"""
function evaluate_objective(coeffs_mat::Matrix{Float64})
    
    # Convert matrix to Dict format
    coeffs_dict = matrix_to_coeffs_dict(coeffs_mat, PROPERTIES, n_modes)
    
    # DEBUG: Print coefficient statistics to verify they're changing
    println("\n[DEBUG] Coefficient stats:")
    println("  μ_l: mean=$(round(mean(coeffs_dict[:μ_l]), digits=3)), std=$(round(std(coeffs_dict[:μ_l]), digits=3)), max_abs=$(round(maximum(abs.(coeffs_dict[:μ_l])), digits=3))")
    println("  μ_t: mean=$(round(mean(coeffs_dict[:μ_t]), digits=3)), std=$(round(std(coeffs_dict[:μ_t]), digits=3)), max_abs=$(round(maximum(abs.(coeffs_dict[:μ_t])), digits=3))")
    
    # Generate fields using pre-computed eigenmodes (EFFICIENT - eigenmodes unchanged!)
    result_fields = Dict{Symbol, Any}()
    
    println("\n  [DEBUG STEP 1] Generating KL material fields from coefficients...")
    for prop_sym in PROPERTIES
        coeffs_for_prop = coeffs_dict[prop_sym]
        println("    Property $prop_sym: coeffs mean=$(round(mean(coeffs_for_prop), digits=3)), std=$(round(std(coeffs_for_prop), digits=3))")
        
        field = sample_KL_field(
            kl_modes_dict[prop_sym], 
            coeffs_for_prop; 
            eltype_out=Float32
        )
        result_fields[prop_sym] = field
        
        # Verify the field is actually different based on coefficients
        println("    Generated field $prop_sym: mean=$(round(mean(field), digits=3)), std=$(round(std(field), digits=3)), min=$(round(minimum(field), digits=3)), max=$(round(maximum(field), digits=3))")
    end
    
    # Add constant properties for any not sampled via KL
    for prop_sym in (:μ_l, :μ_t, :α, :β, :λ, :angle)
        if !haskey(result_fields, prop_sym)
            val = prop_sym == :λ ? mp.λ : (prop_sym == :angle ? mp.angle : getfield(mp, Base.Meta.parse(string(prop_sym))))
            result_fields[prop_sym] = fill(Float32(val), N_ELEM, N_LOC)
        end
    end
    
    # Build material field and validate
    println("\n  [DEBUG STEP 2] Building MaterialField structure...")
    mf = build_material_field(result_fields; use_centroids=false, eltype_out=Float32)
    
    # DEBUG: Check if material fields are varying
    println("    MaterialField.μ_l: mean=$(round(mean(mf.μ_l), digits=3)), std=$(round(std(mf.μ_l), digits=3))")
    println("    MaterialField.μ_t: mean=$(round(mean(mf.μ_t), digits=3)), std=$(round(std(mf.μ_t), digits=3))")
    println("    MaterialField.α: mean=$(round(mean(mf.α), digits=3)), std=$(round(std(mf.α), digits=3))")
    println("    MaterialField.β: mean=$(round(mean(mf.β), digits=3)), std=$(round(std(mf.β), digits=3))")
    
    # Check a specific element to verify spatial variation
    elem_1_μl = [mf.μ_l[1, i] for i in 1:N_LOC]
    elem_1_μt = [mf.μ_t[1, i] for i in 1:N_LOC]
    println("    Element 1 μ_l values across nodes: $(round.(elem_1_μl, digits=2))")
    println("    Element 1 μ_t values across nodes: $(round.(elem_1_μt, digits=2))")

    
    # Validate material field (RELAXED - allow extreme values for adversarial search)
    is_valid, diagnostics = validate_material_field(mf, mp; tolerance_factor=10.0)
    if !is_valid
        # Check for critical failures only (negative shear moduli)
        has_negative_mu = get(diagnostics, "μ_l_negative", false) || get(diagnostics, "μ_t_negative", false)
        
        if has_negative_mu
            @warn "Generated material field with negative shear moduli - returning penalty" diagnostics
            
            # Populate missing keys to prevent KeyError in caller
            diagnostics["intermediate_frac"] = 0.0
            diagnostics["severity"] = 0.0
            diagnostics["gray"] = 0.0
            
            # Return penalty value but don't completely fail
            return -1e3, 1e9, zeros(Float64, N_ELEM), diagnostics
        else
            # Material is extreme but physically valid - proceed
            @info "Material field is extreme but valid" diagnostics
        end
    end
    
    # Build stiffness matrices
    println("  [DEBUG] Rebuilding stiffness matrices with new material field...")
    
    # Save old KE value for comparison (safely check if it's been initialized)
    old_ke_sum = 0.0
    try
        if isdefined(Main, :KE_store) && !isempty(KE_store) && isassigned(KE_store, 1)
            old_ke_sum = sum(abs.(KE_store[1]))
        end
    catch
        # KE_store not yet initialized, that's fine
    end
    
    build_KEStore!(dh, mf, nnodes_loc, avg_mp_store)
    
    # DEBUG: Check if KE_store has changed (it's a vector of matrices)
    try
        if isdefined(Main, :KE_store) && !isempty(KE_store) && isassigned(KE_store, 1)
            first_ke = KE_store[1]
            new_ke_sum = sum(abs.(first_ke))
            println("  [DEBUG] KE_store[1] stats: mean=$(round(mean(abs.(first_ke)), digits=6)), max=$(round(maximum(abs.(first_ke)), digits=3))")
            if old_ke_sum > 0
                println("  [DEBUG] KE_store[1] sum changed: $(round(old_ke_sum, digits=3)) → $(round(new_ke_sum, digits=3)) (diff=$(round(abs(new_ke_sum - old_ke_sum), digits=3)))")
            else
                println("  [DEBUG] KE_store[1] sum: $(round(new_ke_sum, digits=3)) (first build)")
            end
        end
    catch e
        println("  [DEBUG] Could not check KE_store: $e")
    end
    
    # Run topology optimization with error handling
    global u = zeros(ndofs(dh))
    println("  [DEBUG] Starting topology optimization...")
    
    local X, c
    try
        X, c = topopt_run(1)  # Run with ID=1
        println("  [DEBUG] TopOpt converged: c=$c, vol_frac=$(mean(X))")
    catch e
        # Handle domain errors gracefully - save diagnostic data
        if isa(e, DomainError)
            println("\n⚠️  DomainError encountered during topology optimization!")
            println("   Error: $e")
            
            # Save failure diagnostics
            failure_file = joinpath(OUTPUT_ROOT,"failed_topopt_runs.txt")
            open(failure_file, "a") do io
                println(io, "\n" * "="^80)
                println(io, "Failed TopOpt Run - $(Dates.now())")
                println(io, "="^80)
                println(io, "Error: $e")
                println(io, "\nKL Coefficients:")
                for (prop, coeffs) in coeffs_dict
                    println(io, "  $prop: $(coeffs)")
                end
                println(io, "\nMaterial Field Statistics:")
                println(io, "  μ_l: mean=$(mean(mf.μ_l)), std=$(std(mf.μ_l)), min=$(minimum(mf.μ_l)), max=$(maximum(mf.μ_l))")
                println(io, "  μ_t: mean=$(mean(mf.μ_t)), std=$(std(mf.μ_t)), min=$(minimum(mf.μ_t)), max=$(maximum(mf.μ_t))")
                println(io, "  α: mean=$(mean(mf.α)), std=$(std(mf.α)), min=$(minimum(mf.α)), max=$(maximum(mf.α))")
                println(io, "  β: mean=$(mean(mf.β)), std=$(std(mf.β)), min=$(minimum(mf.β)), max=$(maximum(mf.β))")
                if isdefined(Main, :x)
                    println(io, "\nDensity Field x:")
                    println(io, "  mean=$(mean(x)), std=$(std(x)), min=$(minimum(x)), max=$(maximum(x))")
                    println(io, "  First 20 elements: $(x[1:min(20, length(x))])")
                else
                    println(io, "\nDensity Field x: Not yet initialized")
                end
                println(io, "\nStacktrace:")
                println(io, sprint(showerror, e, catch_backtrace()))
                println(io, "="^80)
            end
            
            println("   → Diagnostics saved to: $failure_file")
            println("   → Assigning penalty values and continuing...")
            
            # Add dummy metrics to diagnostics to prevent KeyError downstream
            diagnostics["intermediate_frac"] = 0.0
            diagnostics["severity"] = 0.0
            diagnostics["gray"] = 0.0
            
            # Return penalty values to discourage this parameter region
            # Use very high badness penalty but keep compliance penalty moderate
            return -1000.0, 1e6, zeros(nelx * nely), diagnostics
        else
            # Re-throw non-domain errors
            rethrow(e)
        end
    end
    
    # Compute adversarial metrics
    frac = compute_intermediary_fraction(X)
    severity = compute_intermediary_severity(X)
    gray = compute_gray_indicator(X)
    
    # Combined badness metric (higher = more intermediate densities)
    # Uses adaptive compliance reference for stability (provided by caller if available)
    badness = compute_combined_badness(
        X, c; 
        w_frac=0.4, 
        w_sev=0.4, 
        w_gray=0.2,
        compliance_ref=get(diagnostics, "compliance_ref", c),  # Use adaptive ref if available
        compliance_penalty_threshold=10.0,
        stability_weight=0.15
    )
    
    diagnostics["intermediate_frac"] = frac
    diagnostics["severity"] = severity
    diagnostics["gray"] = gray
    
    return badness, c, X, diagnostics
end

# ============================================================================
# OPTIMIZATION WRAPPER FOR BLACKBOXOPTIM
# ============================================================================

"""
    objective_wrapper(x::Vector{Float64})

Wrapper function for BlackBoxOptim.
Converts vector to matrix, evaluates objective, returns NEGATIVE badness (for minimization).
"""
function objective_wrapper(x::Vector{Float64})
    # Reshape vector to matrix
    max_modes = maximum(values(n_modes))
    coeffs_mat = reshape(x, max_modes, N_PROPS)
    
    # Evaluate (returns badness to MAXIMIZE)
    badness, compliance, X, diagnostics = evaluate_objective(coeffs_mat)
    
    # Return NEGATIVE badness for minimization
    return -badness
end

# ============================================================================
# MAIN OPTIMIZATION FUNCTION
# ============================================================================

"""
    run_adversarial_optimization(; max_iterations=100, population_size=0)

Run adversarial optimization using CMA-ES to find KL coefficients
that maximize intermediate densities in topology optimization.

# Arguments
- `max_iterations`: Maximum number of optimization iterations
- `population_size`: CMA-ES population size (0 = auto)

# Returns
Tuple of (best_coeffs_mat, best_badness, optimization_result)
"""
function run_adversarial_optimization(; max_iterations=100, population_size=0)
    
    println("\n" * "="^80)
    println("STARTING ADVERSARIAL OPTIMIZATION")
    println("="^80)
    
    # Create optimizer configuration
    opt = AdversarialOptimizer(
        n_modes, PROPERTIES;
        max_iterations=max_iterations,
        population_size=population_size,
        initial_sigma=0.5,
        w_frac=0.4,
        w_severity=0.4,
        w_gray=0.2,
        coeff_bounds=(-3.0, 3.0),
        save_path=SAVE_PATH,
        save_every=10
    )
    
    println("\nOptimization Settings:")
    println("  Max iterations:      $(opt.max_iterations)")
    println("  Population size:     $(opt.population_size)")
    println("  Initial sigma:       $(opt.initial_sigma)")
    println("  Coefficient bounds:  [$(opt.coeff_lower_bound), $(opt.coeff_upper_bound)]")
    println("  Total parameters:    $(sum(values(n_modes)))")
    
    # Initialize coefficients
    initial_coeffs = initialize_coefficients(opt; seed=SEED)
    
    # Flatten for BlackBoxOptim
    max_modes = maximum(values(n_modes))
    x0 = vec(initial_coeffs)
    
    # Define search range
    search_range = [(opt.coeff_lower_bound, opt.coeff_upper_bound) for _ in 1:length(x0)]
    
    println("\nStarting CMA-ES optimization...")
    println("="^80)
    
    # Track evaluations for logging AND compliance statistics
    eval_counter = [0]
    compliance_history = Float64[]  # Track compliance values for adaptive reference
    best_badness_so_far = Ref(-Inf)  # Track best badness found
    best_coeffs_so_far = nothing  # Track coefficients of best solution
    best_X_so_far = nothing  # Track density field of best solution
    best_tracker = init_best_tracker()
    
    # Define objective wrapper INSIDE the function (needs access to local variables)
    function objective_wrapper_logged(x::Vector{Float64})
        eval_counter[1] += 1
        iter = eval_counter[1]
        
        # Reshape vector to matrix
        coeffs_mat = reshape(x, max_modes, N_PROPS)
        
        # Evaluate (returns badness to MAXIMIZE)
        badness, compliance, X, diagnostics = evaluate_objective(coeffs_mat)
        
        # Update compliance history for adaptive reference
        push!(compliance_history, compliance)
        
        # Compute adaptive compliance reference (median of observed values)
        # This makes the penalty robust to the actual compliance scale
        compliance_ref = if length(compliance_history) >= 10
            median(compliance_history)
        else
            compliance  # Use current value for first few iterations
        end
        
        # Recompute badness with adaptive compliance reference
        # This ensures stability penalty adapts to actual compliance range
        frac = diagnostics["intermediate_frac"]
        severity = diagnostics["severity"]
        gray = diagnostics["gray"]
        
        badness_adjusted = compute_combined_badness(
            X, compliance;
            w_frac=0.4,
            w_sev=0.4, 
            w_gray=0.2,
            compliance_ref=compliance_ref,
            compliance_penalty_threshold=10.0,  # Penalize if >10x median
            stability_weight=0.15  # Moderate penalty to maintain feasibility
        )
        
        # Update best-so-far tracking
        if badness_adjusted > best_badness_so_far[]
            best_badness_so_far[] = badness_adjusted
            best_coeffs_so_far = copy(coeffs_mat)
            best_X_so_far = copy(X)
            println("  🎯 NEW BEST! Badness=$(round(badness_adjusted, digits=6)) at iteration $iter")
            # Manage VTU: delete previous and save current best
            delete_previous_vtu!(best_tracker)
            save_best_vtu!(best_tracker, X, iter; save_dir=SAVE_PATH)
            # Update tracker fields and persist metadata
            best_tracker.badness = badness_adjusted
            best_tracker.compliance = compliance
            best_tracker.iter = iter
            persist_best_metadata(SAVE_PATH, best_tracker)
        end
        
        # Log this evaluation (use adjusted badness, include best-so-far)
        log_iteration(opt, iter, badness_adjusted, compliance, frac, severity, gray, coeffs_mat, X;
                     best_so_far=best_badness_so_far[])
        
        # Print summary every 10 evaluations
        if iter % 10 == 0 || iter == 1
            print_optimization_summary(iter, badness_adjusted, compliance, frac, severity, gray)
            println("    Best so far: $(round(best_badness_so_far[], digits=6)) (improvement tracking)")
        end
        
        # Save checkpoint periodically
        if iter % opt.save_every == 0
            save_checkpoint(opt, iter, coeffs_mat, X)
        end
        
        # Return NEGATIVE badness for minimization
        return -badness_adjusted
    end
    
    # Run CMA-ES optimization (NO CALLBACK - let optimizer run freely)
    result = bboptimize(
        objective_wrapper_logged;
        SearchRange = search_range,
        NumDimensions = length(x0),
        Method = :adaptive_de_rand_1_bin_radiuslimited,  # Adaptive differential evolution
        MaxFuncEvals = max_iterations,  # Total number of function evaluations
        TraceMode = :compact
    )
    
    # Extract final result
    final_x = best_candidate(result)
    final_coeffs = reshape(final_x, max_modes, N_PROPS)
    final_badness, final_compliance, final_X, final_diag = evaluate_objective(final_coeffs)
    
    # Save final results
    save_final_results(opt, final_coeffs, final_badness)
    
    # Create convergence plot
    create_convergence_plot(opt)
    
    return final_coeffs, final_badness, result
end

# ============================================================================
# MAIN EXECUTION
# ============================================================================

"""
Main execution block - run this to start adversarial optimization.
Adjust parameters below as needed.
"""
function main()
    println("\n" * "="^80)
    println("ADVERSARIAL TOPOLOGY OPTIMIZATION")
    println("Searching for KL coefficients that maximize intermediate densities")
    println("="^80)
    
    # CONFIGURATION - Adjust these as needed
    max_iterations = 50  # Number of optimization iterations
    population_size = 0  # 0 = auto (recommended)
    
    println("\nExecution Parameters:")
    println("  Max iterations: $max_iterations")
    println("  Population size: $(population_size == 0 ? "auto" : population_size)")
    println("  KL modes per property: $N_MODES_PER_PROP")
    println("  Properties optimized: $(join(PROPERTIES, ", "))")
    
    # Run optimization
    try
        coeffs, badness, result = run_adversarial_optimization(
            max_iterations=max_iterations,
            population_size=population_size
        )
        
        println("\n" * "="^80)
        println("OPTIMIZATION COMPLETE!")
        println("="^80)
        println("\nResults saved to: $SAVE_PATH")
        println("\nTo reconstruct material fields from best coefficients:")
        println("  1. Load coefficients from: best_coefficients.txt")
        println("  2. Use `sample_KL_field()` with pre-computed eigenmodes")
        println("  3. Build MaterialField and run TopOpt")
        
        return coeffs, badness, result
        
    catch e
        @error "Optimization failed" exception=e
        rethrow(e)
    end
end

# Run if script is executed directly (not included)
if abspath(PROGRAM_FILE) == @__FILE__
    main()
else
    # Script was included - call main() anyway for interactive use
    println("\n⚠️  Script was included, not executed directly.")
    println("To start optimization, call: main()")
    println("Or re-run with: julia --project=. test/proxy.jl")
end

# Uncomment this line to always run (even when included):
main()
