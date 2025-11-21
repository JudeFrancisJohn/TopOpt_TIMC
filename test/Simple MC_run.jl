using Random
using Statistics
using Dates
using Printf

# Include the main driver file
# This will load all dependencies and define the necessary functions and globals
# It also sets up the initial FE grid and material parameters
println("Loading TopOpt driver...")
include(joinpath(@__DIR__, "..", "src", "COPY_stochastic_modified_v2_MC.jl"))

# --- MCMC Configuration ---
const N_CHAIN = 20          # Number of MCMC iterations
const N_MODES = 80          # Number of KL modes (must match what's used in run_single_design/KL_realization)
const PROPOSAL_SIGMA = 0.5 # Step size for random walk proposal
const BETA = 50.0           # Inverse temperature. Higher = stronger preference for "bad" designs.

# Create a specific output directory for this MCMC chain
dt_str = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
global save_path = joinpath(save_root, "mcmc_chain_$(dt_str)")
mkpath(save_path)
println("Output directory: $save_path")

# --- Helper Functions ---

"""
Evaluate the 'badness' of a design.
Higher score = 'worse' design (which is what we want to find).
"""
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

# --- Initialization ---

println("Initializing MCMC chain...")

# Initial state: random coefficients
# We have 4 properties: :μ_l, :μ_t, :α, :β
current_coeffs = Dict(
    :μ_l => randn(N_MODES),
    :μ_t => randn(N_MODES),
    :α => randn(N_MODES),
    :β => randn(N_MODES)
)

# Run initial sample
println("Running initial sample (Iteration 1)...")
current_log = run_single_design(1, current_coeffs)
current_score = evaluate_badness(current_log)

println("Initial Score: $(round(current_score, digits=4)) (Status: $(current_log.status))")

# Storage for the chain
chain_scores = Float64[]
push!(chain_scores, current_score)
accepted_count = 0

# --- MCMC Loop ---

for i in 2:N_CHAIN
    println("\n--- MCMC Iteration $i / $N_CHAIN ---")

    # 1. Propose new coefficients (Random Walk Metropolis)
    proposal_coeffs = Dict{Symbol,Vector{Float64}}()
    for (k, v) in current_coeffs
        # Add small Gaussian noise to each coefficient
        proposal_coeffs[k] = v + PROPOSAL_SIGMA * randn(N_MODES)
    end

    # 2. Run simulation with proposed coefficients
    println("Running TopOpt for proposal...")
    proposal_log = run_single_design(i, proposal_coeffs)
    proposal_score = evaluate_badness(proposal_log)

    # 3. Metropolis Acceptance Step
    # We want to sample from P(x) ~ exp(BETA * score(x))
    # Acceptance ratio alpha = P(prop) / P(curr) = exp(BETA * (prop_score - curr_score))
    # (Proposal distribution is symmetric, so q factor cancels out)

    score_diff = proposal_score - current_score
    alpha = exp(BETA * score_diff)
    acceptance_prob = min(1.0, alpha)

    println("  Current Score:  $(round(current_score, digits=4))")
    println("  Proposal Score: $(round(proposal_score, digits=4))")
    println("  Score Diff:     $(round(score_diff, digits=4))")
    println("  Acceptance Prob: $(round(acceptance_prob, digits=4))")

    if rand() < acceptance_prob
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
end

# --- Summary ---
println("\n" * "="^50)
println("MCMC Summary")
println("="^50)
println("Total Iterations: $N_CHAIN")
println("Accepted Samples: $accepted_count")
println("Acceptance Rate:  $(round(accepted_count / (N_CHAIN-1), digits=2))")
println("Final Score:      $(round(current_score, digits=4))")
println("Chain Scores:     $chain_scores")
println("Results saved to: $save_path")
