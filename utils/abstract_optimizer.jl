"""
abstract_optimizer.jl

Abstract interface for adversarial optimization strategies.
Implements the Strategy pattern to allow easy switching between different
optimization algorithms (CMA-ES, Simulated Annealing, etc.)
"""

using Random
using Dates
# ============================================================================
# ABSTRACT TYPE DEFINITION
# ============================================================================

"""
    AbstractAdversarialOptimizer

Abstract base type for adversarial optimization strategies.
All concrete optimizers must implement the `optimize!` method.
"""
abstract type AbstractAdversarialOptimizer end

# ============================================================================
# REQUIRED INTERFACE
# ============================================================================

"""
    optimize!(optimizer::AbstractAdversarialOptimizer, objective_fn::Function, initial_coeffs::Matrix{Float64})

Run optimization using the specified strategy.

# Arguments
- `optimizer::AbstractAdversarialOptimizer`: Concrete optimizer instance
- `objective_fn::Function`: Function to optimize, signature `f(coeffs_mat) -> (badness, compliance, X, diagnostics)`
- `initial_coeffs::Matrix{Float64}`: Initial coefficient matrix

# Returns
Tuple of (best_coeffs, best_badness, optimization_result)

# Notes
- This method must be implemented by all concrete optimizer types
- The objective function should return higher values for "better" (more adversarial) solutions
- The optimizer may internally negate the objective for minimization-based algorithms
"""
function optimize!(optimizer::AbstractAdversarialOptimizer, objective_fn::Function, 
                   initial_coeffs::Matrix{Float64})
    error("optimize! not implemented for type $(typeof(optimizer))")
end

# ============================================================================
# SHARED UTILITIES
# ============================================================================

"""
    initialize_coefficients(n_modes::Dict, properties::Tuple; seed::Int=42, initial_sigma::Float64=0.5)

Initialize coefficient matrix with random values from normal distribution.

# Arguments
- `n_modes::Dict{Symbol,Int}`: Number of modes per property
- `properties::Tuple`: Property symbols
- `seed::Int`: Random seed for reproducibility
- `initial_sigma::Float64`: Standard deviation for initialization

# Returns
Matrix of size (max_modes, n_props) with random coefficients
"""
function initialize_coefficients(n_modes::Dict{Symbol,Int}, properties::Tuple; 
                                  seed::Int=42, initial_sigma::Float64=0.5)
    Random.seed!(seed)
    max_modes = maximum(values(n_modes))
    n_props = length(properties)
    
    coeffs_mat = randn(max_modes, n_props) .* initial_sigma
    
    # Zero out unused modes
    for (j, prop) in enumerate(properties)
        n = n_modes[prop]
        if n < max_modes
            coeffs_mat[(n+1):end, j] .= 0.0
        end
    end
    
    return coeffs_mat
end

"""
    flatten_coeffs(coeffs_mat::Matrix{Float64})

Flatten coefficient matrix to vector for optimization.
"""
function flatten_coeffs(coeffs_mat::Matrix{Float64})
    return vec(coeffs_mat)
end

"""
    unflatten_coeffs(coeffs_vec::Vector{Float64}, max_modes::Int, n_props::Int)

Reshape coefficient vector to matrix.
"""
function unflatten_coeffs(coeffs_vec::Vector{Float64}, max_modes::Int, n_props::Int)
    return reshape(coeffs_vec, max_modes, n_props)
end

# ============================================================================
# COMMON CONFIGURATION STRUCT
# ============================================================================

"""
    OptimizerConfig

Common configuration parameters shared across all optimizer types.
"""
struct OptimizerConfig
    # Problem dimensions
    n_modes::Dict{Symbol, Int}
    properties::Tuple
    max_modes::Int
    n_props::Int
    
    # Bounds
    coeff_lower_bound::Float64
    coeff_upper_bound::Float64
    
    # Objective weights
    w_frac::Float64
    w_severity::Float64
    w_gray::Float64
    w_stability::Float64
    
    # Logging
    save_path::String
    save_every::Int
    
    # Misc
    seed::Int
    
    function OptimizerConfig(n_modes::Dict{Symbol,Int}, properties::Tuple;
                            coeff_bounds::Tuple{Float64,Float64}=(-3.0, 3.0),
                            w_frac::Float64=0.4,
                            w_severity::Float64=0.4,
                            w_gray::Float64=0.2,
                            w_stability::Float64=0.15,
                            save_path::String="",
                            save_every::Int=10,
                            seed::Int=42)
        
        max_modes = maximum(values(n_modes))
        n_props = length(properties)
        
        if isempty(save_path)

            dt_str = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
            save_path = joinpath("output", "adversarial_$(dt_str)")
        end
        mkpath(save_path)
        
        new(n_modes, properties, max_modes, n_props,
            coeff_bounds[1], coeff_bounds[2],
            w_frac, w_severity, w_gray, w_stability,
            save_path, save_every, seed)
    end
end
