using Random
using Statistics

# ==============================================================================
# 1. Define the Target Distribution
# ==============================================================================
"""
    log_target_density(params)

Calculates the log-probability density of the target distribution at `params`.
We work with logs to avoid numerical underflow.

For this example, the target is a 2D Gaussian centered at [0, 0].
"""
function log_target_density(params::Vector{Float64})
    # Example: Target is proportional to exp(-0.5 * x^2) (Standard Normal)
    # log(exp(...)) simplifies to just the exponent.
    return -0.5 * sum(params.^2)
end

# ==============================================================================
# 2. Define the Proposal Mechanism
# ==============================================================================
"""
    propose_new_state(current_params, step_size)

Generates a candidate state by adding random Gaussian noise to the current state.
This is a "Random Walk Metropolis" proposal.
"""
function propose_new_state(current_params::Vector{Float64}, step_size::Float64)
    noise = randn(length(current_params)) * step_size
    return current_params + noise
end

# ==============================================================================
# 3. The Metropolis-Hastings Algorithm
# ==============================================================================
"""
    run_metropolis_hastings(initial_params, n_samples, step_size)

Runs the MCMC chain.
- `initial_params`: Starting point in parameter space.
- `n_samples`: Number of iterations to run.
- `step_size`: Standard deviation of the proposal noise.
"""
function run_metropolis_hastings(initial_params::Vector{Float64}, n_samples::Int, step_size::Float64)

    n_params = length(initial_params)
    chain_samples = zeros(Float64, n_samples, n_params)

    current_params = initial_params
    current_log_prob = log_target_density(current_params)
    accepted_count = 0
    

    for i in 1:n_samples

        proposed_params = propose_new_state(current_params, step_size)
        proposed_log_prob = log_target_density(proposed_params)
        
        # B. Calculate Acceptance Ratio (in Log Space)
        log_acceptance_ratio = proposed_log_prob - current_log_prob
        
        # C. Accept or Reject
        # We accept if log(U) < log_acceptance_ratio, where U ~ Uniform(0,1)
        if log(rand()) < log_acceptance_ratio
            # Update state
            current_params = proposed_params
            current_log_prob = proposed_log_prob
            accepted_count += 1
        else
            # Reject: current_params remains the same
        end
        
        # Store the state (whether it changed or not)
        chain_samples[i, :] = current_params
    end
    
    acceptance_rate = accepted_count / n_samples
    println("MCMC Finished.")
    println("  Acceptance Rate: $(round(acceptance_rate * 100, digits=2))%")
    println("  (Ideal acceptance rate is usually between 20% and 50%)")
    
    return chain_samples
end

"""function main()
    # Setup
    initial_guess = [10.0, -10.0]  # Start far away from the center [0,0]
    num_samples = 10_000
    proposal_step_size = 0.5       # Tuning parameter: too big = low acceptance, too small = slow mixing

    # Run
    chain = run_metropolis_hastings(initial_guess, num_samples, proposal_step_size)

    # Simple Analysis
    mean_estimate = mean(chain, dims=1)
    println("\nEstimated Mean (should be close to [0.0, 0.0]):")
    println(mean_estimate)
end
"""
