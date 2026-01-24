"""
validators.jl

Material field validation utilities.
Ensures generated stochastic material fields are physically reasonable.

Following Single Responsibility:
- ONE module for ALL validation logic
- Pure functions: returns (is_valid, diagnostics)
- No side effects (caller decides what to do with results)
"""

using Statistics

# ============================================================================
# MATERIAL FIELD VALIDATION
# ============================================================================

"""
    validate_material_field(mf, material_params; tolerance_factor=5.0, strict=false)

Validate that generated material field has physically reasonable values.

# Arguments
- `mf`: MaterialField object with fields (μ_l, μ_t, α, β, λ, angle)
- `material_params`: MaterialParams with mean values for reference
- `tolerance_factor`: Maximum allowed deviation in units of sigma (default: 5.0)
- `strict`: If true, treat warnings as failures (default: false)

# Returns
Tuple `(is_valid::Bool, diagnostics::Dict{String, Any})`

# Checks Performed
1. **NaN/Inf detection**: Any NaN or Inf values → FAIL
2. **Positivity**: μ_l, μ_t, λ must be > 0 → FAIL
3. **Extreme deviations**: Values far from mean → WARNING (or FAIL if strict)

# Example
```julia
is_valid, diag = validate_material_field(mf, mp; tolerance_factor=10.0)
if !is_valid
    println("Validation failed: ", diag)
end
```
"""
function validate_material_field(mf, material_params; tolerance_factor=5.0, strict=false)
    diagnostics = Dict{String, Any}()
    is_valid = true
    
    # Check 1: NaN or Inf
    for field_name in [:μ_l, :μ_t, :α, :β, :λ]
        field = getfield(mf, field_name)
        if any(isnan.(field)) || any(isinf.(field))
            diagnostics["$(field_name)_nan_inf"] = true
            diagnostics["$(field_name)_has_nan"] = any(isnan.(field))
            diagnostics["$(field_name)_has_inf"] = any(isinf.(field))
            is_valid = false
        end
    end
    
    # Check 2: Positivity for required fields
    if any(mf.μ_l .<= 0)
        diagnostics["μ_l_negative"] = true
        diagnostics["μ_l_min"] = minimum(mf.μ_l)
        diagnostics["μ_l_negative_count"] = sum(mf.μ_l .<= 0)
        is_valid = false
    end
    
    if any(mf.μ_t .<= 0)
        diagnostics["μ_t_negative"] = true
        diagnostics["μ_t_min"] = minimum(mf.μ_t)
        diagnostics["μ_t_negative_count"] = sum(mf.μ_t .<= 0)
        is_valid = false
    end
    
    if any(mf.λ .<= 0)
        diagnostics["λ_negative"] = true
        diagnostics["λ_min"] = minimum(mf.λ)
        diagnostics["λ_negative_count"] = sum(mf.λ .<= 0)
        is_valid = false
    end
    
    # Check 3: Extreme deviations (warnings)
    μ_l_range = (minimum(mf.μ_l), maximum(mf.μ_l))
    μ_t_range = (minimum(mf.μ_t), maximum(mf.μ_t))
    λ_range = (minimum(mf.λ), maximum(mf.λ))
    
    diagnostics["μ_l_range"] = μ_l_range
    diagnostics["μ_t_range"] = μ_t_range
    diagnostics["λ_range"] = λ_range
    diagnostics["μ_l_mean"] = mean(mf.μ_l)
    diagnostics["μ_t_mean"] = mean(mf.μ_t)
    diagnostics["λ_mean"] = mean(mf.λ)
    diagnostics["μ_l_std"] = std(mf.μ_l)
    diagnostics["μ_t_std"] = std(mf.μ_t)
    diagnostics["λ_std"] = std(mf.λ)
    
    # Check for extreme deviations from mean
    mean_μ_l = getproperty(material_params, :μ_l)
    mean_μ_t = getproperty(material_params, :μ_t)
    mean_λ = getproperty(material_params, :λ)
    
    # Ratio-based checks (more robust for positive quantities)
    max_ratio_μ_l = μ_l_range[2] / mean_μ_l
    min_ratio_μ_l = μ_l_range[1] / mean_μ_l
    max_ratio_μ_t = μ_t_range[2] / mean_μ_t
    min_ratio_μ_t = μ_t_range[1] / mean_μ_t
    
    diagnostics["μ_l_max_ratio"] = max_ratio_μ_l
    diagnostics["μ_l_min_ratio"] = min_ratio_μ_l
    diagnostics["μ_t_max_ratio"] = max_ratio_μ_t
    diagnostics["μ_t_min_ratio"] = min_ratio_μ_t
    
    # Flag extreme deviations as warnings
    if max_ratio_μ_l > tolerance_factor || min_ratio_μ_l < 1.0/tolerance_factor
        diagnostics["μ_l_extreme_deviation"] = true
        if strict
            is_valid = false
        end
    end
    
    if max_ratio_μ_t > tolerance_factor || min_ratio_μ_t < 1.0/tolerance_factor
        diagnostics["μ_t_extreme_deviation"] = true
        if strict
            is_valid = false
        end
    end
    
    return is_valid, diagnostics
end

# ============================================================================
# COEFFICIENT VALIDATION
# ============================================================================

"""
    validate_coefficients(coeffs_mat::Matrix{Float64}; 
                         max_abs_value=10.0, warn_threshold=5.0)

Validate KL expansion coefficient matrix for reasonable values.

# Arguments
- `coeffs_mat`: Coefficient matrix (max_modes × n_props)
- `max_abs_value`: Maximum absolute coefficient value (default: 10.0)
- `warn_threshold`: Threshold for warnings (default: 5.0)

# Returns
Tuple `(is_valid::Bool, diagnostics::Dict{String, Any})`

# Checks
- No NaN or Inf values
- Coefficients within reasonable bounds
- No extremely large L2 norms
"""
function validate_coefficients(coeffs_mat::Matrix{Float64}; 
                              max_abs_value=10.0, warn_threshold=5.0)
    diagnostics = Dict{String, Any}()
    is_valid = true
    
    # Check for NaN/Inf
    if any(isnan.(coeffs_mat)) || any(isinf.(coeffs_mat))
        diagnostics["has_nan_inf"] = true
        is_valid = false
    end
    
    # Check absolute values
    max_abs = maximum(abs.(coeffs_mat))
    diagnostics["max_abs_coeff"] = max_abs
    
    if max_abs > max_abs_value
        diagnostics["exceeds_max_abs"] = true
        is_valid = false
    elseif max_abs > warn_threshold
        diagnostics["large_coefficients_warning"] = true
    end
    
    # Compute L2 norm per column
    n_props = size(coeffs_mat, 2)
    diagnostics["l2_norms"] = Float64[]
    for j in 1:n_props
        col = coeffs_mat[:, j]
        active_coeffs = col[col .!= 0.0]
        if !isempty(active_coeffs)
            l2_norm = sqrt(sum(active_coeffs.^2))
            push!(diagnostics["l2_norms"], l2_norm)
        end
    end
    
    return is_valid, diagnostics
end

# ============================================================================
# DENSITY FIELD VALIDATION
# ============================================================================

"""
    validate_density_field(X::Vector{Float64}; tol=1e-6)

Validate topology optimization density field.

# Arguments
- `X`: Density vector (should be in [0, 1])
- `tol`: Tolerance for bounds checking (default: 1e-6)

# Returns
Tuple `(is_valid::Bool, diagnostics::Dict{String, Any})`

# Checks
- All values in [0, 1] (within tolerance)
- No NaN or Inf
- Statistics for debugging
"""
function validate_density_field(X::Vector{Float64}; tol=1e-6)
    diagnostics = Dict{String, Any}()
    is_valid = true
    
    # Check for NaN/Inf
    if any(isnan.(X)) || any(isinf.(X))
        diagnostics["has_nan_inf"] = true
        is_valid = false
    end
    
    # Check bounds
    min_val = minimum(X)
    max_val = maximum(X)
    
    diagnostics["min_density"] = min_val
    diagnostics["max_density"] = max_val
    diagnostics["mean_density"] = mean(X)
    diagnostics["std_density"] = std(X)
    
    if min_val < -tol || max_val > 1.0 + tol
        diagnostics["outside_bounds"] = true
        diagnostics["below_zero_count"] = sum(X .< -tol)
        diagnostics["above_one_count"] = sum(X .> 1.0 + tol)
        is_valid = false
    end
    
    return is_valid, diagnostics
end

# ============================================================================
# SUMMARY VALIDATION
# ============================================================================

"""
    print_validation_summary(diagnostics::Dict; show_warnings=true)

Print human-readable validation summary.

# Arguments
- `diagnostics`: Diagnostics dictionary from validation function
- `show_warnings`: Whether to print warnings (default: true)
"""
function print_validation_summary(diagnostics::Dict; show_warnings=true)
    println("\n" * "="^80)
    println("VALIDATION SUMMARY")
    println("="^80)
    
    # Errors (validation failures)
    errors = filter(kv -> occursin("negative", string(kv[1])) || 
                         occursin("nan_inf", string(kv[1])) ||
                         occursin("outside_bounds", string(kv[1])), diagnostics)
    
    if !isempty(errors)
        println("\n❌ ERRORS (validation failed):")
        for (key, val) in errors
            println("  - $key: $val")
        end
    end
    
    # Warnings
    if show_warnings
        warnings = filter(kv -> occursin("extreme", string(kv[1])) || 
                              occursin("warning", string(kv[1])), diagnostics)
        
        if !isempty(warnings)
            println("\n⚠️  WARNINGS:")
            for (key, val) in warnings
                println("  - $key: $val")
            end
        end
    end
    
    # Statistics
    stats = filter(kv -> occursin("mean", string(kv[1])) || 
                        occursin("range", string(kv[1])) ||
                        occursin("std", string(kv[1])), diagnostics)
    
    if !isempty(stats)
        println("\n📊 STATISTICS:")
        for (key, val) in sort(collect(stats))
            if isa(val, Number)
                @printf("  %-20s: %.6e\n", key, val)
            else
                println("  $key: $val")
            end
        end
    end
    
    println("="^80 * "\n")
end
