using Random
using Statistics
using Dates
using Printf

#Include the main driver file
#This will load all dependencies and define the necessary functions and globals
#It also sets up the initial FE grid and material parameters
println("Loading TopOpt driver...")
include( "../src/COPY_stochastic_modified_v2_MC.jl")
include("../input/params_MCMC.jl")
# Import the new diagnostic utilities
include(joinpath(@__DIR__, "..", "utils", "diagnostic_utils.jl"))
using .DiagnosticUtils: initialize_diagnostics!, monitor_diagnostics!
include(joinpath(@__DIR__, "..", "utils", "mcmc_utils.jl"))




# Create a specific output directory for this MCMC chain
const OUTPUT_ROOT = normpath(joinpath(@__DIR__, "..", "output"))
global save_root = OUTPUT_ROOT
dt_str = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
global save_path = joinpath(save_root, "mcmc_chain_$(dt_str)")
chain_file = normpath(joinpath(save_root, "mcmc_chain_$(dt_str).txt"))
println("Output file: $(chain_file)")

if isdefined(Main, :WRITE_OUTPUT_FILES) && WRITE_OUTPUT_FILES
    error("WRITE_OUTPUT_FILES must remain false to prevent heavy VTK output for this run.")
end
mkpath(dirname(chain_file))

#

extract_density_log(log_entry) = hasproperty(log_entry, :final_density_log) ? getfield(log_entry, :final_density_log) : (isa(log_entry, AbstractDict) && haskey(log_entry, :final_density_log) ? log_entry[:final_density_log] : nothing)




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
# Initialize diagnostics for tracked metrics
params_to_monitor = [:badness, :acceptance_prob]
initialize_diagnostics!(params_to_monitor)

# Record diagnostics for the initial sample
initial_params = Dict(:badness => current_score, :acceptance_prob => 1.0)
monitor_diagnostics!(initial_params, 1.0, 1, 10)

# --- MCMC Loop ---

post_burnin_scores = Float64[]
io = open(chain_file, "w")
try
    timestamp = Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS")
    println(io, "# Simple MC MCMC results saved at $(timestamp)")
    println(io, "# Configuration")
    println(io, "N_CHAIN=$(N_CHAIN)")
    println(io, "BURN_IN=$(BURN_IN)")
    println(io, "N_MODES=$(N_MODES)")
    println(io, "PROPOSAL_SIGMA=$(PROPOSAL_SIGMA)")
    println(io, "BETA=$(BETA)")
    println(io, "DENSITY_WEIGHT=$(DENSITY_WEIGHT)")
    println(io, "COMPLIANCE_WEIGHT=$(COMPLIANCE_WEIGHT)")
    println(io)
    println(io, "# Iteration Details")
    write_iteration_record(io, 1;
        accepted=true,
        is_burnin=1 <= BURN_IN,
        current_score=current_score,
        proposal_score=current_score,
        acceptance_prob=1.0,
        running_acceptance=1.0,
        log_entry=current_log,
        coeffs=current_coeffs)
    flush(io)

    for i in 2:N_CHAIN
        println("\n--- MCMC Iteration $i / $N_CHAIN ---")

        proposal_coeffs = Dict{Symbol,Vector{Float64}}()
        for (k, v) in current_coeffs
            proposal_coeffs[k] = v + PROPOSAL_SIGMA * randn(N_MODES)
        end

        println("Running TopOpt for proposal...")
        proposal_log = run_single_design(i, proposal_coeffs)
        proposal_score = evaluate_badness(proposal_log)

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
        end

        running_acceptance = accepted_count / (i - 1)
        current_params = Dict(:badness => current_score, :acceptance_prob => acceptance_prob)
        monitor_diagnostics!(current_params, running_acceptance, i, 10)

        push!(chain_scores, current_score)
        write_iteration_record(io, i;
            accepted=accepted,
            is_burnin=i <= BURN_IN,
            current_score=current_score,
            proposal_score=proposal_score,
            acceptance_prob=acceptance_prob,
            running_acceptance=running_acceptance,
            log_entry=current_log,
            coeffs=current_coeffs)
        flush(io)
    end

    post_burnin_scores = BURN_IN < length(chain_scores) ? chain_scores[BURN_IN+1:end] : Float64[]
    println(io, "# Final Summary")
    println(io, "Accepted_Samples=$(accepted_count)")
    println(io, "Acceptance_Rate=$(round(accepted_count / max(1, N_CHAIN-1), digits=4))")
    println(io, "Final_Score=$(round(current_score, digits=6))")
    chain_scores_str = join(string.(round.(chain_scores, digits=6)), ", ")
    post_burnin_str = join(string.(round.(post_burnin_scores, digits=6)), ", ")
    println(io, "Chain_Scores=$(chain_scores_str)")
    println(io, "Post_Burnin_Scores=$(post_burnin_str)")
finally
    close(io)
end

println("\n" * "="^50)
println("MCMC Summary")
println("="^50)
println("Total Iterations: $N_CHAIN")
println("Burn-in: $BURN_IN")
println("Accepted Samples: $accepted_count")
println("Acceptance Rate:  $(round(accepted_count / max(1, N_CHAIN-1), digits=2))")
println("Final Score:      $(round(current_score, digits=4))")
println("Chain Scores:     $chain_scores")
println("Post Burn-in Scores: $post_burnin_scores")
println("Results saved to file: $chain_file")
