using Random
using Statistics
using Dates
using Printf
using JLD2

# Include the main driver file
# This will load all dependencies and define the necessary functions and globals
# It also sets up the initial FE grid and material parameters
println("Loading TopOpt driver...")
include( "../src/COPY_stochastic_modified_v2_MC.jl")
include("../utils/mcmc_utils.jl")


# --- MCMC Configuration ---
const N_CHAIN = 20          # Number of MCMC iterations
const BURN_IN = Int(floor(0.3 * N_CHAIN))  # Burn-in iterations discarded from analysis
const N_MODES = 80          # Number of KL modes (must match what's used in run_single_design/KL_realization)
const PROPOSAL_SIGMA = 0.5 # Step size for random walk proposal
const BETA = 50.0           # Inverse temperature. Higher = stronger preference for "bad" designs.
const DENSITY_WEIGHT = 5.0
const COMPLIANCE_WEIGHT = 1.0e-3

# KL Expansion parameters (must match between MCMC and MC runs)
const KL_SIGMA_DICT = Dict(
    :μ_l => 0.5,
    :μ_t => 0.5,
    :α => 0.5,
    :β => 0.5
)
const KL_LC = 0.01
const KL_KERNEL = :exponential
const KL_MODE = :lognormal
const KL_USE_CENTROIDS = false
const KL_MAKE_SPARSE = true

# Create a specific output directory for this MCMC chain
dt_str = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
global save_path = joinpath(save_root, "mcmc_chain_$(dt_str)")
chain_file = joinpath(save_root, "mcmc_chain_$(dt_str).jld2")
println("Output file: $chain_file")

function evaluate_badness(log_entry)
    # If the run failed, return a score of 0.0 (or very low).
    if log_entry.status != :success
        return 0.0
    end

    props = if hasproperty(log_entry, :final_density_log)
        getfield(log_entry, :final_density_log)
    elseif isa(log_entry, AbstractDict) && haskey(log_entry, :final_density_log)
        log_entry[:final_density_log]
    else
        return 0.0
    end

    bin_props = if hasproperty(props, :bin_proportions)
        getfield(props, :bin_proportions)
    elseif isa(props, AbstractDict) && haskey(props, :bin_proportions)
        props[:bin_proportions]
    else
        return 0.0
    end

    # Focus on intermediary density (bin 2)
    mid_prop = bin_props[2]
    mid_density_term = DENSITY_WEIGHT * mid_prop

    # Compliance term (optional, very small weight)
    comp = get_compliance(log_entry)
    compliance_term = if comp === nothing
        0.0
    else
        COMPLIANCE_WEIGHT * comp
    end

    # Total badness: prioritize intermediary density
    return mid_density_term + compliance_term
end

"""
Run a topology optimization with material fields generated from KL coefficients.
Uses pre-computed eigenmodes for efficiency.
"""
function run_with_kl_coeffs(run_i::Int, coeffs_dict::Dict{Symbol,Vector{Float64}}, kl_modes_dict::Dict{Symbol,KL_Eigenmodes})
    # Generate material fields from pre-computed eigenmodes and provided coefficients
    fields = Dict{Symbol,Any}()
    for prop_sym in [:μ_l, :μ_t, :α, :β]
        kl_modes = kl_modes_dict[prop_sym]
        coeffs = coeffs_dict[prop_sym]
        fields[prop_sym] = sample_KL_field(kl_modes, coeffs; eltype_out=Float32)
    end
    
    # Add constant fields for λ and angle
    n_elem, n_loc = size(coords_elem)
    fields[:λ] = KL_USE_CENTROIDS ? fill(mp.λ, n_elem) : fill(mp.λ, n_elem, n_loc)
    fields[:angle] = KL_USE_CENTROIDS ? fill(mp.angle, n_elem) : fill(mp.angle, n_elem, n_loc)
    
    # Compute statistics for logging
    stats = shear_stats_from_fields(fields)
    
    # Build material field structure
    mf = build_material_field(fields; use_centroids=KL_USE_CENTROIDS, eltype_out=Float32)
    
    # Update stiffness matrices
    build_KEStore!(dh, mf, nnodes_loc, avg_mp_store)
    
    # Reset global displacement vector
    global u = zeros(ndofs(dh))
    
    # Run topology optimization
    log_entry = topopt_run(run_i, stats, 0, nothing)
    
    return log_entry
end

# ============================================================================
# PRE-COMPUTE KL EIGENMODES (DETERMINISTIC - DONE ONCE)
# ============================================================================

println("\n" * "="^80)
println("║" * " "^20 * "PRE-COMPUTING KL EXPANSION EIGENMODES" * " "^20 * "║")
println("="^80)
println("This is the DETERMINISTIC part of KL expansion - solving eigenvalue problems")
println("based on mean material values, covariance structure, and geometry.")
println("These eigenmodes will be FIXED throughout the MCMC chain.\n")

kl_modes_dict = Dict{Symbol,KL_Eigenmodes}()

for (prop_sym, sigma) in KL_SIGMA_DICT
    print("  Computing eigenmodes for $(prop_sym)...")
    flush(stdout)
    
    kl_modes = compute_KL_eigenmodes(mp, coords_elem, prop_sym, sigma;
        Lc=KL_LC, 
        N_modes=N_MODES,
        use_centroids=KL_USE_CENTROIDS, 
        make_sparse=KL_MAKE_SPARSE, 
        eltype_out=Float32,
        kernel=KL_KERNEL, 
        mode=KL_MODE)
    
    kl_modes_dict[prop_sym] = kl_modes
    n_computed = length(kl_modes.eigenvalues)
    println(" ✓ ($(n_computed) modes)")
end

println("\n✓ All eigenmodes pre-computed successfully!")
println("  From now on, MCMC will only sample coefficients (ξᵢ) for these fixed eigenmodes.\n")
println("="^80)

# --- Initialization ---

println("\n" * "="^80)
println("║" * " "^25 * "INITIALIZING MCMC CHAIN" * " "^25 * "║")
println("="^80)

# Initial state: random coefficients (ξᵢ ~ N(0,1))
# We have 4 properties: :μ_l, :μ_t, :α, :β
current_coeffs = Dict(
    :μ_l => randn(N_MODES),
    :μ_t => randn(N_MODES),
    :α => randn(N_MODES),
    :β => randn(N_MODES)
)

# Run initial sample
println("\nRunning initial sample (Iteration 1)...")
current_log = run_with_kl_coeffs(1, current_coeffs, kl_modes_dict)
current_score = evaluate_badness(current_log)

println("Initial Score: $(round(current_score, digits=4)) (Status: $(current_log.status))")

# Storage for the chain
chain_scores = Float64[]
push!(chain_scores, current_score)
accepted_count = 0
chain_records = Vector{Dict{Symbol,Any}}()
push!(chain_records, Dict(
    :iteration => 1,
    :accepted => true,
    :score => current_score,
    :acceptance_prob => 1.0,
    :coeffs => deepcopy(current_coeffs),
    :log_entry => current_log,
    :is_burnin => 1 <= BURN_IN))

println("="^80)

# --- MCMC Loop ---

println("\n" * "="^80)
println("║" * " "^26 * "STARTING MCMC SAMPLING" * " "^27 * "║")
println("="^80)
println("Exploring coefficient space to find material parameters that lead to")
println("designs with high intermediary densities (bad designs).\n")

for i in 2:N_CHAIN
    println("\n--- MCMC Iteration $i / $N_CHAIN ---")

    # 1. Propose new coefficients (Random Walk Metropolis)
    # Only the coefficients change - eigenmodes stay fixed!
    proposal_coeffs = Dict{Symbol,Vector{Float64}}()
    for (k, v) in current_coeffs
        # Add small Gaussian noise to each coefficient
        proposal_coeffs[k] = v + PROPOSAL_SIGMA * randn(N_MODES)
    end

    # 2. Run simulation with proposed coefficients using FIXED eigenmodes
    println("Running TopOpt for proposal...")
    proposal_log = run_with_kl_coeffs(i, proposal_coeffs, kl_modes_dict)
    proposal_score = evaluate_badness(proposal_log)

    # 3. Metropolis Acceptance Step
    # We want to sample from P(x) ~ exp(BETA * score(x))
    # Acceptance ratio alpha = P(prop) / P(curr) = exp(BETA * (prop_score - curr_score))
    # (Proposal distribution is symmetric, so q factor cancels out)

    score_diff = proposal_score - current_score
    log_acceptance = BETA * score_diff
    acceptance_prob = log_acceptance >= 0 ? 1.0 : exp(log_acceptance)

    println("  Current Score:  $(round(current_score, digits=4))")
    println("  Proposal Score: $(round(proposal_score, digits=4))")
    println("  Score Diff:     $(round(score_diff, digits=4))")
    println("  Acceptance Prob: $(round(acceptance_prob, digits=4))")

    accepted = rand() < acceptance_prob
    if accepted
        println("  -> ACCEPTED")
        global current_coeffs = proposal_coeffs
        global current_score = proposal_score
        global current_log = proposal_log
        global accepted_count += 1
    else
        println("  -> REJECTED")
        # We stay at current_coeffs
    end

    push!(chain_scores, current_score)
    push!(chain_records, Dict(
        :iteration => i,
        :accepted => accepted,
        :score => current_score,
        :acceptance_prob => acceptance_prob,
        :coeffs => deepcopy(current_coeffs),
        :log_entry => current_log,
        :is_burnin => i <= BURN_IN))
end

# --- Summary ---
println("\n" * "="^80)
println("║" * " "^28 * "MCMC CHAIN COMPLETE" * " "^29 * "║")
println("="^80)
println("Total Iterations: $N_CHAIN")
println("Burn-in: $BURN_IN")
println("Accepted Samples: $accepted_count")
println("Acceptance Rate:  $(round(accepted_count / (N_CHAIN-1), digits=2))")
println("Final Score:      $(round(current_score, digits=4))")
post_burnin_scores = BURN_IN < length(chain_scores) ? chain_scores[BURN_IN+1:end] : Float64[]
println("Chain Scores:     $chain_scores")
println("Post Burn-in Scores: $post_burnin_scores")

# Save results with KL expansion configuration
@save chain_file chain_records chain_scores post_burnin_scores accepted_count PROPOSAL_SIGMA BETA N_CHAIN BURN_IN DENSITY_WEIGHT COMPLIANCE_WEIGHT KL_SIGMA_DICT KL_LC KL_KERNEL KL_MODE N_MODES

println("\n✓ Results saved to file: $chain_file")
println("="^80)
