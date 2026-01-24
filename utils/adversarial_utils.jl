"""
adversarial_utils.jl

Coefficient format conversion utilities for adversarial optimization.

Following Single Responsibility Principle:
- Handles ONLY coefficient format conversions (matrix ↔ dictionary)
- Metrics moved to metrics.jl
- Validation moved to validators.jl
- I/O moved to io_manager.jl

This module is now focused solely on data structure transformations.
"""

using Printf
using Statistics
using LinearAlgebra
using Dates

# ============================================================================
# COEFFICIENT FORMAT CONVERSION
# ============================================================================

"""
    matrix_to_coeffs_dict(coeffs_mat::Matrix{Float64}, properties::Tuple, n_modes::Dict{Symbol,Int})

Convert coefficient matrix to dictionary format for KL field generation.
Each column represents coefficients for one material property.

# Arguments
- `coeffs_mat`: Matrix of size (max_modes, n_props)
- `properties`: Tuple of property symbols (e.g., (:μ_l, :μ_t, :α, :β))
- `n_modes`: Dictionary mapping property symbols to number of modes

# Returns
Dictionary mapping property symbols to coefficient vectors
"""
function matrix_to_coeffs_dict(coeffs_mat::Matrix{Float64}, properties::Tuple, n_modes::Dict{Symbol,Int})
    coeffs_dict = Dict{Symbol, Vector{Float64}}()
    for (j, prop) in enumerate(properties)
        n = n_modes[prop]
        coeffs_dict[prop] = coeffs_mat[1:n, j]
    end
    return coeffs_dict
end

"""
    coeffs_dict_to_matrix(coeffs_dict::Dict{Symbol,Vector{Float64}}, properties::Tuple, max_modes::Int)

Convert coefficient dictionary to matrix format for optimization.
Returns a matrix where each column represents one property's coefficients.

# Arguments
- `coeffs_dict`: Dictionary mapping property symbols to coefficient vectors
- `properties`: Tuple of property symbols
- `max_modes`: Maximum number of modes across all properties

# Returns
Matrix of size (max_modes, n_props)
"""
function coeffs_dict_to_matrix(coeffs_dict::Dict{Symbol,Vector{Float64}}, properties::Tuple, max_modes::Int)
    n_props = length(properties)
    coeffs_mat = zeros(Float64, max_modes, n_props)
    for (j, prop) in enumerate(properties)
        n = length(coeffs_dict[prop])
        coeffs_mat[1:n, j] .= coeffs_dict[prop]
    end
    return coeffs_mat
end

# ============================================================================
# COEFFICIENT ANALYSIS
# ============================================================================



"""
    compute_coefficient_stats(coeffs_mat::Matrix{Float64}, properties::Tuple)

Compute statistics about coefficient distribution.

# Arguments
- `coeffs_mat`: Coefficient matrix (max_modes × n_props)
- `properties`: Tuple of property symbols

# Returns
Dictionary with statistics for each property
"""
function compute_coefficient_stats(coeffs_mat::Matrix{Float64}, properties::Tuple)
    stats = Dict{Symbol, Dict{String, Float64}}()
    
    for (j, prop) in enumerate(properties)
        coeffs = coeffs_mat[:, j]
        # Filter out zeros (unused modes)
        active_coeffs = coeffs[coeffs .!= 0.0]
        
        stats[prop] = Dict(
            "mean" => mean(active_coeffs),
            "std" => std(active_coeffs),
            "max_abs" => maximum(abs.(active_coeffs)),
            "n_active" => length(active_coeffs),
            "norm_l2" => norm(active_coeffs, 2),
            "norm_linf" => norm(active_coeffs, Inf)
        )
    end
    
    return stats
end

"""
    export_coefficients_to_txt(coeffs_mat::Matrix{Float64}, properties::Tuple, 
                               n_modes::Dict{Symbol,Int}, filepath::String;
                               include_stats=true)

Export coefficient matrix to human-readable text file.

# Arguments
- `coeffs_mat`: Coefficient matrix
- `properties`: Tuple of property symbols
- `n_modes`: Dictionary of active modes per property
- `filepath`: Output file path
- `include_stats`: Whether to include statistics
"""
function export_coefficients_to_txt(coeffs_mat::Matrix{Float64}, properties::Tuple, 
                                   n_modes::Dict{Symbol,Int}, filepath::String;
                                   include_stats=true)
    open(filepath, "w") do io
        println(io, "# KL Expansion Coefficients")
        println(io, "# Generated: $(Dates.now())")
        println(io, "# Properties: $(join(properties, ", "))")
        println(io, "#" ^ 80)
        println(io)
        
        if include_stats
            println(io, "# STATISTICS")
            stats = compute_coefficient_stats(coeffs_mat, properties)
            for prop in properties
                println(io, "\n## $prop")
                for (key, val) in sort(collect(stats[prop]))
                    @printf(io, "  %-12s: %.6e\n", key, val)
                end
            end
            println(io, "\n" * "#" ^ 80)
            println(io)
        end
        
        println(io, "# COEFFICIENTS (mode_index, property, value)")
        for (j, prop) in enumerate(properties)
            n = n_modes[prop]
            println(io, "\n## $prop ($(n) modes)")
            for i in 1:n
                @printf(io, "%4d  %8s  %16.10e\n", i, prop, coeffs_mat[i, j])
            end
        end
    end
    
    println("Coefficients exported to: $filepath")
end

"""
    print_optimization_summary(iteration::Int, badness::Float64, compliance::Float64,
                              frac::Float64, severity::Float64, gray::Float64)

Print formatted summary of optimization iteration.
"""
function print_optimization_summary(iteration::Int, badness::Float64, compliance::Float64,
                                   frac::Float64, severity::Float64, gray::Float64)
    println("=" ^ 80)
    @printf("Iteration %3d\n", iteration)
    println("-" ^ 80)
    @printf("  Badness:              %.6f\n", badness)
    @printf("  Compliance:           %.6e\n", compliance)
    @printf("  Intermediate Frac:    %.4f (%.1f%%)\n", frac, frac*100)
    @printf("  Severity:             %.6f\n", severity)
    @printf("  Gray Indicator:       %.6f\n", gray)
    println("=" ^ 80)
end
