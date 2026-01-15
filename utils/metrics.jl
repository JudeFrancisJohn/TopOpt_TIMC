"""
metrics.jl

Adversarial optimization metrics for topology optimization.
Computes various measures of intermediate density "badness" in SIMP-based designs.

Following DRY and Single Responsibility:
- ONE module for ALL metric computations
- Pure functions: input → output (no I/O)
- Composable: individual metrics + combined metric
"""

using Statistics

# ============================================================================
# INDIVIDUAL METRICS
# ============================================================================

"""
    compute_intermediary_fraction(X::Vector{Float64};
                                 threshold_low=0.1,
                                  threshold_high=0.9)

Compute fraction of design variables with intermediate densities.
Higher values indicate more intermediate densities (worse for SIMP).

# Arguments
- `X`: Design density vector
- `threshold_low`: Lower threshold for intermediate region (default: 0.1)
- `threshold_high`: Upper threshold for intermediate region (default: 0.9)

# Returns
Fraction of elements in intermediate density range [threshold_low, threshold_high]

# Example
```julia
X = [0.0, 0.3, 0.5, 0.8, 1.0]
frac = compute_intermediary_fraction(X)  # Returns 0.6 (3 out of 5)
```
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
- `threshold_low`: Lower threshold for intermediate region (default: 0.1)
- `threshold_high`: Upper threshold for intermediate region (default: 0.9)

# Returns
Average severity score (0.0 to 0.5, where 0.5 is worst)

# Details
For each intermediate element, computes min(x, 1-x) and averages.
Elements at x=0.5 have maximum severity (0.5).
Elements near boundaries (0.1 or 0.9) have lower severity (~0.1).
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

Compute "gray" indicator metric from topology optimization literature.
Measures deviation from binary (0/1) solution.

# Formula
GI = (4/n) * Σ x_i * (1 - x_i)

# Arguments
- `X`: Design density vector

# Returns
Gray indicator value:
- 0.0 = fully binary solution (all 0 or 1)
- 1.0 = maximally gray (all at 0.5)

# References
Sigmund, O. (2007). "Morphology-based black and white filters for topology optimization"
"""
function compute_gray_indicator(X::Vector{Float64})
    n = length(X)
    return (4.0 / n) * sum(X .* (1.0 .- X))
end

# ============================================================================
# STABILITY PENALTY
# ============================================================================

"""
    compute_stability_penalty(compliance::Float64, compliance_ref::Float64;
                             penalty_threshold=100.0, weight=0.1)

Compute penalty for extreme compliance values indicating numerical instability.

# Arguments
- `compliance`: Current compliance value
- `compliance_ref`: Reference compliance for normalization
- `penalty_threshold`: Threshold ratio for triggering penalty (default: 100.0)
- `weight`: Weight of penalty in final badness (default: 0.1)

# Returns
Penalty value (0.0 if stable, positive if unstable)

# Details
Uses soft exponential thresholding to penalize:
- Very high compliance (approaching singularity)
- Very low compliance (potentially degenerate)
"""
function compute_stability_penalty(compliance::Float64, compliance_ref::Float64;
                                  penalty_threshold=100.0, weight=0.1)
    compliance_ratio = compliance / compliance_ref
    
    if compliance_ratio > penalty_threshold
        # Exponential penalty for very high compliance (approaching instability)
        penalty = weight * (1.0 - exp(-(compliance_ratio - penalty_threshold) / 50.0))
    elseif compliance_ratio < 1.0 / penalty_threshold
        # Also penalize very low compliance (might indicate degenerate solutions)
        penalty = weight * (1.0 - exp(-(1.0/compliance_ratio - penalty_threshold) / 50.0))
    else
        # No penalty in reasonable compliance range
        penalty = 0.0
    end
    
    return penalty
end

# ============================================================================
# COMBINED BADNESS METRIC
# ============================================================================

"""
    compute_combined_badness(X::Vector{Float64}, compliance::Float64; 
                            w_frac=0.4, w_sev=0.4, w_gray=0.2,
                            compliance_ref=1.0, 
                            compliance_penalty_threshold=100.0,
                            stability_weight=0.1)

Compute combined badness metric for adversarial optimization.
Combines multiple intermediate density metrics with stability penalty.

# Arguments
- `X`: Design density vector
- `compliance`: Compliance value from TopOpt run
- `w_frac`: Weight for intermediate fraction (default: 0.4)
- `w_sev`: Weight for severity (default: 0.4)
- `w_gray`: Weight for gray indicator (default: 0.2)
- `compliance_ref`: Reference compliance for normalization (default: 1.0)
- `compliance_penalty_threshold`: Threshold for stability penalty (default: 100.0)
- `stability_weight`: Weight for stability penalty (default: 0.1)

# Returns
Combined badness score (higher = more intermediate densities = more adversarial)

# Formula
badness = w_frac * frac + w_sev * severity_norm + w_gray * gray - stability_penalty

where severity_norm = severity / 0.5 (normalizes to [0,1])

# Notes
- Weights should sum to 1.0 for interpretability
- Stability penalty is subtracted (reduces badness for unstable solutions)
- Use adaptive compliance_ref (e.g., median of history) for better convergence
"""
function compute_combined_badness(X::Vector{Float64}, compliance::Float64; 
                                 w_frac=0.4, w_sev=0.4, w_gray=0.2,
                                 compliance_ref=1.0, 
                                 compliance_penalty_threshold=100.0,
                                 stability_weight=0.1)
    # Compute individual metrics
    frac = compute_intermediary_fraction(X)
    severity = compute_intermediary_severity(X)
    gray = compute_gray_indicator(X)
    
    # Normalize severity to 0-1 (max severity is 0.5)
    severity_norm = severity / 0.5
    
    # Base badness from intermediate densities
    badness_raw = w_frac * frac + w_sev * severity_norm + w_gray * gray
    
    # Stability penalty
    stability_penalty = compute_stability_penalty(
        compliance, compliance_ref;
        penalty_threshold=compliance_penalty_threshold,
        weight=stability_weight
    )
    
    # Final badness: high is good, but penalized if approaching instability
    badness = badness_raw - stability_penalty
    
    return badness
end

# ============================================================================
# METRIC DECOMPOSITION (for analysis)
# ============================================================================

"""
    compute_all_metrics(X::Vector{Float64}, compliance::Float64; 
                       compliance_ref=1.0, kwargs...)

Compute all metrics and return as named tuple for detailed analysis.

# Returns
NamedTuple with fields:
- `frac`: Intermediate fraction
- `severity`: Severity score
- `gray`: Gray indicator
- `stability_penalty`: Stability penalty
- `badness_raw`: Badness without stability penalty
- `badness`: Final combined badness

# Example
```julia
metrics = compute_all_metrics(X, compliance)
println("Fraction: ", metrics.frac)
println("Final badness: ", metrics.badness)
```
"""
function compute_all_metrics(X::Vector{Float64}, compliance::Float64; 
                            compliance_ref=1.0,
                            w_frac=0.4, w_sev=0.4, w_gray=0.2,
                            compliance_penalty_threshold=100.0,
                            stability_weight=0.1)
    frac = compute_intermediary_fraction(X)
    severity = compute_intermediary_severity(X)
    gray = compute_gray_indicator(X)
    severity_norm = severity / 0.5
    
    badness_raw = w_frac * frac + w_sev * severity_norm + w_gray * gray
    
    stability_penalty = compute_stability_penalty(
        compliance, compliance_ref;
        penalty_threshold=compliance_penalty_threshold,
        weight=stability_weight
    )
    
    badness = badness_raw - stability_penalty
    
    return (
        frac=frac,
        severity=severity,
        severity_norm=severity_norm,
        gray=gray,
        stability_penalty=stability_penalty,
        badness_raw=badness_raw,
        badness=badness
    )
end
