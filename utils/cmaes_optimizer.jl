"""
cmaes_optimizer.jl

CMA-ES (Covariance Matrix Adaptation Evolution Strategy) implementation
for adversarial coefficient optimization.

Uses BlackBoxOptim.jl for the actual optimization.
"""

using BlackBoxOptim
using Statistics
using Printf

include("abstract_optimizer.jl")
include("io_manager.jl")
include("metrics.jl")
include("adversarial_utils.jl")

# ============================================================================
# CMA-ES OPTIMIZER
# ============================================================================

"""
    CMAESOptimizer

CMA-ES based adversarial optimizer using BlackBoxOptim.jl.
Focuses purely on optimization state - all I/O handled by IOManager.
"""
mutable struct CMAESOptimizer <: AbstractAdversarialOptimizer
    config::OptimizerConfig
    max_iterations::Int
    population_size::Int
    initial_sigma::Float64
    bboptim_method::Symbol
    
    # Algorithm-specific tracking (not I/O)
    compliance_history::Vector{Float64}
    best_badness_ref::Ref{Float64}
    best_coeffs_ref::Ref{Union{Nothing, Matrix{Float64}}}
    best_X_ref::Ref{Union{Nothing, Vector{Float64}}}
    
    function CMAESOptimizer(config::OptimizerConfig;
                           max_iterations::Int=100,
                           population_size::Int=0,
                           initial_sigma::Float64=0.5,
                           bboptim_method::Symbol=:adaptive_de_rand_1_bin_radiuslimited)
        
        # Auto-determine population size if needed
        n_params = sum(values(config.n_modes))
        if population_size == 0
            population_size = 4 + floor(Int, 3 * log(n_params))
        end
        
        new(config, max_iterations, population_size, initial_sigma, bboptim_method,
            Float64[], Ref(-Inf), Ref{Union{Nothing, Matrix{Float64}}}(nothing), Ref{Union{Nothing, Vector{Float64}}}(nothing))
    end
end

"""
    optimize!(optimizer::CMAESOptimizer, objective_fn::Function, initial_coeffs::Matrix{Float64}, export_vtk_fn::Function, io_manager::IOManager)

Run CMA-ES optimization with clean separation of concerns.

# Arguments
- `optimizer::CMAESOptimizer`: Optimizer instance (holds algorithm state)
- `objective_fn::Function`: Objective function with signature `f(coeffs_mat) -> (badness, compliance, X, diagnostics)`
- `initial_coeffs::Matrix{Float64}`: Initial coefficient matrix
- `export_vtk_fn::Function`: VTK export function
- `io_manager::IOManager`: Handles all I/O operations (printing, saving, exporting)

# Returns
Tuple of (best_coeffs_mat, best_badness, bboptimize_result)
"""
function optimize!(optimizer::CMAESOptimizer, objective_fn::Function, 
                   initial_coeffs::Matrix{Float64}, export_vtk_fn::Function,
                   io_manager::IOManager)
    
    config = optimizer.config
    
    # Print header via IOManager
    print_header(io_manager, config.properties, config.n_modes, 
                 optimizer.max_iterations, "CMA-ES")
    
    # Flatten initial coefficients for BlackBoxOptim
    x0 = flatten_coeffs(initial_coeffs)
    search_range = [(config.coeff_lower_bound, config.coeff_upper_bound) for _ in 1:length(x0)]
    
    # Evaluation counter
    eval_counter = [0]
    
    # Create objective wrapper for BlackBoxOptim
    function objective_wrapper(x::Vector{Float64})
        eval_counter[1] += 1
        iter = eval_counter[1]

        coeffs_mat = unflatten_coeffs(x, config.max_modes, config.n_props)
        badness, compliance, X, diagnostics = objective_fn(coeffs_mat)
        push!(optimizer.compliance_history, compliance)
        
        # Compute adaptive compliance reference (median of observed values)
        compliance_ref = if length(optimizer.compliance_history) >= 10
            median(optimizer.compliance_history)
        else
            compliance
        end
        
        # Recompute badness with adaptive compliance reference
        badness_adjusted = compute_combined_badness(
            X, compliance;
            w_frac=config.w_frac,
            w_sev=config.w_severity,
            w_gray=config.w_gray,
            compliance_ref=compliance_ref,
            compliance_penalty_threshold=10.0,
            stability_weight=config.w_stability
        )
        
        # Update diagnostics with adjusted values
        diagnostics["badness_adjusted"] = badness_adjusted
        
        # Update best-so-far tracking
        if badness_adjusted > optimizer.best_badness_ref[]
            old_best = optimizer.best_badness_ref[]
            optimizer.best_badness_ref[] = badness_adjusted
            optimizer.best_coeffs_ref[] = copy(coeffs_mat)
            optimizer.best_X_ref[] = copy(X)
            
            # Delegate all I/O to IOManager
            handle_new_best(io_manager, badness_adjusted, compliance, iter, X, 
                          coeffs_mat, export_vtk_fn)
        end
        
        # Log iteration to history (via IOManager)
        log_iteration(io_manager, iter, badness_adjusted, compliance, diagnostics)
        
        # Print summary periodically (via IOManager)
        if should_print_summary(iter)
            print_iteration_summary(io_manager, iter, badness_adjusted, compliance, 
                                   diagnostics, optimizer.best_badness_ref[])
        end
        
        # Save checkpoint periodically (via IOManager)
        if should_save_checkpoint(io_manager, iter)
            save_checkpoint(io_manager, iter, coeffs_mat, X)
        end
        
        # Return NEGATIVE badness for minimization
        return -badness_adjusted
    end
    
    # Run BlackBoxOptim (pure optimization, no I/O)
    result = bboptimize(
        objective_wrapper;
        SearchRange = search_range,
        NumDimensions = length(x0),
        Method = optimizer.bboptim_method,
        MaxFuncEvals = optimizer.max_iterations,
        TraceMode = :compact
    )
    
    # Extract final result from BlackBoxOptim
    final_x = best_candidate(result)
    final_coeffs = unflatten_coeffs(final_x, config.max_modes, config.n_props)
    
    # Use the best coefficients found during optimization (tracked internally)
    # This is more reliable than re-evaluating the BlackBoxOptim result
    best_coeffs_found = if optimizer.best_coeffs_ref[] !== nothing
        optimizer.best_coeffs_ref[]
    else
        # Fallback: use final BlackBoxOptim candidate
        final_coeffs
    end
    
    best_badness_found = if optimizer.best_badness_ref[] > -Inf
        optimizer.best_badness_ref[]
    else
        # Fallback: evaluate the final solution
        badness, _, _, _ = objective_fn(final_coeffs)
        badness
    end
    
    # Save final results (via IOManager)
    save_final_results(io_manager, best_coeffs_found, best_badness_found; 
                      properties=optimizer.config.properties, 
                      n_modes=optimizer.config.n_modes)
    
    # Create convergence plot
    create_convergence_plot(io_manager)
    print_completion_summary(io_manager, best_badness_found)
    
    return best_coeffs_found, best_badness_found, result
end
