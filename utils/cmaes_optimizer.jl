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
include("adversarial_logging.jl")
include("adversarial_utils.jl")

# ============================================================================
# CMA-ES OPTIMIZER
# ============================================================================

"""
    CMAESOptimizer

CMA-ES based adversarial optimizer using BlackBoxOptim.jl.
"""
mutable struct CMAESOptimizer <: AbstractAdversarialOptimizer
    config::OptimizerConfig
    max_iterations::Int
    population_size::Int
    initial_sigma::Float64
    
    # Tracking
    history::Dict{String, Vector{Any}}
    best_tracker::BestVTUTracker
    compliance_history::Vector{Float64}
    best_badness_ref::Ref{Float64}
    best_coeffs_ref::Ref{Union{Nothing, Matrix{Float64}}}
    best_X_ref::Ref{Union{Nothing, Vector{Float64}}}
    
    function CMAESOptimizer(config::OptimizerConfig;
                           max_iterations::Int=100,
                           population_size::Int=0,
                           initial_sigma::Float64=0.5)
        
        # Auto-determine population size if needed
        n_params = sum(values(config.n_modes))
        if population_size == 0
            population_size = 4 + floor(Int, 3 * log(n_params))
        end
        
        # Initialize history
        history = Dict{String, Vector{Any}}(
            "iteration" => Int[],
            "badness" => Float64[],
            "compliance" => Float64[],
            "intermediate_frac" => Float64[],
            "severity" => Float64[],
            "gray_indicator" => Float64[],
            "timestamp" => String[]
        )
        
        best_tracker = init_best_tracker()
        
        new(config, max_iterations, population_size, initial_sigma,
            history, best_tracker, Float64[],
            Ref(-Inf), Ref(nothing), Ref(nothing))
    end
end

"""
    optimize!(optimizer::CMAESOptimizer, objective_fn::Function, initial_coeffs::Matrix{Float64})

Run CMA-ES optimization.

# Arguments
- `optimizer::CMAESOptimizer`: Optimizer instance
- `objective_fn::Function`: Objective function with signature `f(coeffs_mat) -> (badness, compliance, X, diagnostics)`
- `initial_coeffs::Matrix{Float64}`: Initial coefficient matrix

# Returns
Tuple of (best_coeffs_mat, best_badness, bboptimize_result)
"""
function optimize!(optimizer::CMAESOptimizer, objective_fn::Function, 
                   initial_coeffs::Matrix{Float64}, export_vtk_fn::Function)
    
    config = optimizer.config
    
    println_optimization_header(config.save_path, config.properties, config.n_modes, 
                                 optimizer.max_iterations)
    
    # Flatten initial coefficients
    x0 = flatten_coeffs(initial_coeffs)
    search_range = [(config.coeff_lower_bound, config.coeff_upper_bound) for _ in 1:length(x0)]
    
    println("\nStarting CMA-ES optimization...")
    println("="^80)
    
    # Evaluation counter
    eval_counter = [0]
    
    # Create objective wrapper
    function objective_wrapper(x::Vector{Float64})
        eval_counter[1] += 1
        iter = eval_counter[1]
        
        # Reshape to matrix
        coeffs_mat = unflatten_coeffs(x, config.max_modes, config.n_props)
        
        # Evaluate objective
        badness, compliance, X, diagnostics = objective_fn(coeffs_mat)
        
        # Track compliance history for adaptive reference
        push!(optimizer.compliance_history, compliance)
        
        # Compute adaptive compliance reference (median of observed values)
        compliance_ref = if length(optimizer.compliance_history) >= 10
            median(optimizer.compliance_history)
        else
            compliance
        end
        
        # Recompute badness with adaptive compliance reference
        frac = diagnostics["intermediate_frac"]
        severity = diagnostics["severity"]
        gray = diagnostics["gray"]
        
        badness_adjusted = compute_combined_badness(
            X, compliance;
            w_frac=config.w_frac,
            w_sev=config.w_severity,
            w_gray=config.w_gray,
            compliance_ref=compliance_ref,
            compliance_penalty_threshold=10.0,
            stability_weight=config.w_stability
        )
        
        # Update best-so-far tracking
        if badness_adjusted > optimizer.best_badness_ref[]
            optimizer.best_badness_ref[] = badness_adjusted
            optimizer.best_coeffs_ref[] = copy(coeffs_mat)
            optimizer.best_X_ref[] = copy(X)
            
            print_best_update(badness_adjusted, iter, optimizer.best_badness_ref[])
            
            # Manage VTU files
            delete_previous_vtu!(optimizer.best_tracker)
            
            # Create wrapper for export function
            function export_wrapper(save_dir, filename_prefix; density=nothing)
                export_vtk_fn(save_dir, filename_prefix, density)
            end
            
            save_best_vtu!(optimizer.best_tracker, X, iter, config.save_path, export_wrapper)
            
            optimizer.best_tracker.badness = badness_adjusted
            optimizer.best_tracker.compliance = compliance
            optimizer.best_tracker.iter = iter
            persist_best_metadata(config.save_path, optimizer.best_tracker)
        end
        
        # Log iteration
        log_iteration_to_history!(optimizer.history, iter, badness_adjusted, compliance, 
                                   frac, severity, gray)
        
        # Print summary periodically
        if iter % 10 == 0 || iter == 1
            print_optimization_summary(iter, badness_adjusted, compliance, frac, severity, gray)
            println("    Best so far: $(round(optimizer.best_badness_ref[], digits=6))")
        end
        
        # Save checkpoint periodically
        if iter % config.save_every == 0
            save_checkpoint(config.save_path, iter, coeffs_mat, X)
        end
        
        # Return NEGATIVE badness for minimization
        return -badness_adjusted
    end
    
    # Run BlackBoxOptim
    result = bboptimize(
        objective_wrapper;
        SearchRange = search_range,
        NumDimensions = length(x0),
        Method = :adaptive_de_rand_1_bin_radiuslimited,
        MaxFuncEvals = optimizer.max_iterations,
        TraceMode = :compact
    )
    
    # Extract final result
    final_x = best_candidate(result)
    final_coeffs = unflatten_coeffs(final_x, config.max_modes, config.n_props)
    
    # Evaluate final solution
    final_badness, final_compliance, final_X, final_diag = objective_fn(final_coeffs)
    
    # Save final results
    save_final_results(config.save_path, final_coeffs, final_badness, optimizer.history)
    
    return final_coeffs, final_badness, result
end

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

function log_iteration_to_history!(history::Dict, iter::Int, badness::Float64, 
                                    compliance::Float64, frac::Float64, 
                                    severity::Float64, gray::Float64)
    using Dates
    push!(history["iteration"], iter)
    push!(history["badness"], badness)
    push!(history["compliance"], compliance)
    push!(history["intermediate_frac"], frac)
    push!(history["severity"], severity)
    push!(history["gray_indicator"], gray)
    push!(history["timestamp"], string(Dates.now()))
end

function save_checkpoint(save_path::String, iter::Int, coeffs_mat::Matrix{Float64}, 
                         X::Vector{Float64})
    using JLD2
    checkpoint_file = joinpath(save_path, "checkpoint_iter_$(iter).jld2")
    jldsave(checkpoint_file; coefficients=coeffs_mat, density_field=X, iteration=iter)
    println("    💾 Saved checkpoint: $(checkpoint_file)")
end

function save_final_results(save_path::String, coeffs_mat::Matrix{Float64}, 
                            badness::Float64, history::Dict)
    using JLD2
    
    # Save coefficients
    coeffs_file = joinpath(save_path, "best_coefficients.txt")
    open(coeffs_file, "w") do io
        println(io, "Best Coefficients (badness = $(badness))")
        println(io, "Shape: $(size(coeffs_mat))")
        println(io, "\nMatrix:")
        for i in 1:size(coeffs_mat, 1)
            println(io, join(coeffs_mat[i, :], ", "))
        end
    end
    
    # Save history
    history_file = joinpath(save_path, "optimization_history.jld2")
    jldsave(history_file; history=history)
    
    println("\n✅ Final results saved:")
    println("   Coefficients: $(coeffs_file)")
    println("   History: $(history_file)")
end

function println_optimization_header(save_path::String, properties::Tuple, 
                                     n_modes::Dict, max_iterations::Int)
    print_optimization_header(save_path, properties, n_modes, max_iterations)
    println("\nOptimizer: CMA-ES (Adaptive Differential Evolution)")
end
