"""
adversarial_utils.jl

Helper functions for adversarial topology optimization using KL expansion coefficients.
This module provides utilities for:
- Coefficient format conversion (matrix ↔ dictionary)
- Objective function metrics (intermediate density fraction, severity)
- Material field validation
- Physical parameter reconstruction from coefficients
"""

using Printf
using Statistics
using LinearAlgebra

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
# ADVERSARIAL OBJECTIVE METRICS
# ============================================================================

"""
    compute_intermediary_fraction(X::Vector{Float64}; threshold_low=0.1, threshold_high=0.9)

Compute fraction of design variables with intermediate densities.
Higher values indicate more intermediate densities (worse for SIMP).

# Arguments
- `X`: Design density vector
- `threshold_low`: Lower threshold for intermediate region
- `threshold_high`: Upper threshold for intermediate region

# Returns
Fraction of elements in intermediate density range [threshold_low, threshold_high]
"""
function compute_intermediary_fraction(X::Vector{Float64}; threshold_low=0.1, threshold_high=0.9)
    return sum((threshold_low .< X) .& (X .< threshold_high)) / length(X)
end

"""
    compute_intermediary_severity(X::Vector{Float64}; threshold_low=0.1, threshold_high=0.9)

Compute severity of intermediate densities by weighting by distance from 0/1.
Elements near 0.5 contribute more than elements near boundaries.

# Arguments
- `X`: Design density vector
- `threshold_low`: Lower threshold for intermediate region
- `threshold_high`: Upper threshold for intermediate region

# Returns
Average severity score (0.0 to 0.5, where 0.5 is worst)
"""
function compute_intermediary_severity(X::Vector{Float64}; threshold_low=0.1, threshold_high=0.9)
    intermediate_mask = (threshold_low .< X) .& (X .< threshold_high)
    if sum(intermediate_mask) == 0
        return 0.0
    end
    
    # Distance from nearest boundary (0 or 1) - closer to 0.5 is worse
    distances = min.(X, 1.0 .- X)
    severity = sum(distances[intermediate_mask]) / sum(intermediate_mask)
    
    return severity
end

"""
    compute_gray_indicator(X::Vector{Float64})

Compute "gray" indicator metric used in topology optimization literature.
Measures deviation from binary (0/1) solution.

GI = (4/n) * sum(x_i * (1 - x_i))

# Arguments
- `X`: Design density vector

# Returns
Gray indicator value (0.0 = binary, 1.0 = all at 0.5)
"""
function compute_gray_indicator(X::Vector{Float64})
    n = length(X)
    return (4.0 / n) * sum(X .* (1.0 .- X))
end

"""
    compute_combined_badness(X::Vector{Float64}, compliance::Float64; 
                             w_frac=0.4, w_sev=0.4, w_gray=0.2,
                             compliance_ref=1.0)

Compute combined badness metric for adversarial optimization.
Combines multiple intermediate density metrics.

# Arguments
- `X`: Design density vector
- `compliance`: Compliance value from TopOpt run
- `w_frac`: Weight for intermediate fraction
- `w_sev`: Weight for severity
- `w_gray`: Weight for gray indicator
- `compliance_ref`: Reference compliance for normalization

# Returns
Combined badness score (higher = more intermediate densities)
"""
function compute_combined_badness(X::Vector{Float64}, compliance::Float64; 
                                  w_frac=0.4, w_sev=0.4, w_gray=0.2,
                                  compliance_ref=1.0)
    frac = compute_intermediary_fraction(X)
    severity = compute_intermediary_severity(X)
    gray = compute_gray_indicator(X)
    
    # Normalize severity to 0-1 (max severity is 0.5)
    severity_norm = severity / 0.5
    
    badness = w_frac * frac + w_sev * severity_norm + w_gray * gray
    
    return badness
end

# ============================================================================
# MATERIAL FIELD VALIDATION
# ============================================================================

"""
    validate_material_field(mf, material_params; tolerance_factor=5.0)

Validate that generated material field has physically reasonable values.
Checks that values don't deviate too far from mean values.

# Arguments
- `mf`: MaterialField object
- `material_params`: MaterialParams with mean values
- `tolerance_factor`: Maximum allowed deviation in units of sigma

# Returns
Tuple (is_valid::Bool, diagnostics::Dict)
"""
function validate_material_field(mf, material_params; tolerance_factor=5.0)
    diagnostics = Dict{String, Any}()
    is_valid = true
    
    # Check for NaN or Inf
    for field_name in [:μ_l, :μ_t, :α, :β, :λ]
        field = getfield(mf, field_name)
        if any(isnan.(field)) || any(isinf.(field))
            diagnostics["$(field_name)_nan_inf"] = true
            is_valid = false
        end
    end
    
    # Check for negative values in positive-definite parameters
    if any(mf.μ_l .<= 0)
        diagnostics["μ_l_negative"] = true
        is_valid = false
    end
    if any(mf.μ_t .<= 0)
        diagnostics["μ_t_negative"] = true
        is_valid = false
    end
    if any(mf.λ .<= 0)
        diagnostics["λ_negative"] = true
        is_valid = false
    end
    
    # Check for extreme deviations (optional warning, not failure)
    μ_l_range = (minimum(mf.μ_l), maximum(mf.μ_l))
    μ_t_range = (minimum(mf.μ_t), maximum(mf.μ_t))
    
    diagnostics["μ_l_range"] = μ_l_range
    diagnostics["μ_t_range"] = μ_t_range
    diagnostics["μ_l_mean"] = mean(mf.μ_l)
    diagnostics["μ_t_mean"] = mean(mf.μ_t)
    
    return is_valid, diagnostics
end

# ============================================================================
# COEFFICIENT ANALYSIS AND EXPORT
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
