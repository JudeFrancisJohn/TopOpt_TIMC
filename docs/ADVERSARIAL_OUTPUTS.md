# Adversarial Optimization Outputs Guide

## Overview

This document describes all outputs recorded during adversarial optimization runs.

---

## Output Directory Structure

```
output/adversarial_YYYYMMDD_HHMMSS/
├── optimization_log.txt              # Text log of iteration objectives
├── best_coefficients.txt             # Best KL coefficients found (human-readable)
├── final_coefficients.txt            # Final iteration coefficients (human-readable)
├── best_density_field.jld2           # Best density field X (from best iteration)
├── optimization_history.jld2         # Complete data (metrics + best/final coeffs)
└── convergence.png                   # Convergence plots (4 panels)
```

**Note**: No checkpoint subdirectory needed - all iteration objectives are in `optimization_log.txt`!

---

## Per-Iteration Metrics Recorded

### 1. **Iteration Number**
- **Variable**: `iteration`
- **Type**: Integer
- **Description**: Sequential iteration counter (1, 2, 3, ...)

### 2. **Badness Score** ⭐ (Primary Objective)
- **Variable**: `badness`
- **Type**: Float64
- **Range**: 0.0 to ~1.0
- **Description**: Combined adversarial objective (higher = more intermediate densities)
- **Formula**: 
  ```
  badness = 0.4 × frac + 0.4 × (severity/0.5) + 0.2 × gray
  ```
- **Goal**: MAXIMIZE this value

### 3. **Compliance**
- **Variable**: `compliance`
- **Type**: Float64
- **Units**: Dimensionless (scaled by loads/material properties)
- **Description**: Structural compliance (inverse of stiffness)
- **Note**: This is the inner TopOpt objective (minimized by SIMP), NOT directly optimized by outer loop

### 4. **Intermediate Fraction**
- **Variable**: `intermediate_frac`
- **Type**: Float64
- **Range**: 0.0 to 1.0
- **Description**: Fraction of elements with densities in range [0.1, 0.9]
- **Example**: 0.194 means 19.4% of elements are "gray" (intermediate)

### 5. **Severity**
- **Variable**: `severity`
- **Type**: Float64
- **Range**: 0.0 to 0.5
- **Description**: Average distance from 0/1 for intermediate densities
- **Interpretation**: 
  - 0.0 = No intermediate densities OR all near boundaries (0.1 or 0.9)
  - 0.5 = All intermediate densities at worst point (0.5)

### 6. **Gray Indicator**
- **Variable**: `gray_indicator`
- **Type**: Float64
- **Range**: 0.0 to 1.0
- **Description**: Standard topology optimization metric
- **Formula**: `GI = (4/n) × sum(x_i × (1 - x_i))`
- **Interpretation**:
  - 0.0 = Perfect binary solution (all 0s or 1s)
  - 1.0 = Worst case (all densities at 0.5)

### 7. **KL Coefficients**
- **Variable**: `coeffs`
- **Type**: Matrix{Float64} (max_modes × n_props)
- **Description**: KL expansion coefficients that generated this design
- **Properties**: (:μ_l, :μ_t, :α, :β)
- **Number of modes per property**: Typically 10-20

### 8. **Timestamp**
- **Variable**: `timestamp`
- **Type**: String
- **Format**: ISO 8601 (e.g., "2025-12-01T14:23:45.123")
- **Description**: When this iteration completed

---

## File Formats

### `optimization_log.txt` (Text Log)

**Format**: Space-delimited columns with headers

**Example**:
```
# Adversarial Optimization Log
# Started: 2025-12-01T14:00:00.000
# Max iterations: 50
# Population size: 14
# Number of parameters: 40
####################################################################################################
Iter     Badness      Compliance   Frac         Severity     Gray        
----------------------------------------------------------------------------------------------------
1        0.341857     2.892346e+00 0.194200     0.287867     0.169485    
2        0.356421     3.015234e+00 0.208500     0.295123     0.178234    
3        0.378945     3.145678e+00 0.225600     0.310456     0.192345    
...
```

**Columns**:
1. `Iter`: Iteration number
2. `Badness`: Combined objective (to maximize)
3. `Compliance`: Structural compliance
4. `Frac`: Intermediate density fraction
5. `Severity`: Intermediate density severity
6. `Gray`: Gray indicator metric

---

### `best_coefficients.txt` (Human-Readable Coefficients)

**Format**: Text with statistics and coefficient values

**Example**:
```
# KL Expansion Coefficients
# Generated: 2025-12-01T15:30:00.000
# Properties: μ_l, μ_t, α, β
################################################################################

# STATISTICS

## μ_l
  max_abs     : 2.345678e+00
  mean        : -1.234567e-02
  n_active    : 15
  norm_l2     : 3.456789e+00
  norm_linf   : 2.345678e+00
  std         : 8.901234e-01

## μ_t
  ...

################################################################################

# COEFFICIENTS (mode_index, property, value)

## μ_l (15 modes)
   1      μ_l      1.2345678901e+00
   2      μ_l     -8.7654321098e-01
   3      μ_l      5.4321098765e-01
   ...
  15      μ_l     -1.2345678901e-01

## μ_t (15 modes)
   1      μ_t      9.8765432109e-01
   ...
```

**Statistics Explained**:
- `max_abs`: Largest coefficient magnitude (check for extremes)
- `mean`: Average coefficient value (should be near 0)
- `n_active`: Number of non-zero modes used
- `norm_l2`: L2 norm (overall energy in coefficients)
- `norm_linf`: L∞ norm (max absolute value)
- `std`: Standard deviation of coefficients

---

### `best_density_field.jld2` (Best Topology Design)

**Format**: JLD2 (Julia binary format)

**How to Load**:
```julia
using JLD2
density_data = load("output/adversarial_YYYYMMDD_HHMMSS/best_density_field.jld2")
```

**Contents**:
```julia
Dict with keys:
  "X"         => Vector{Float64}  # Density field (n_elements values, 0-1 range)
  "iteration" => Int              # Which iteration produced this
  "badness"   => Float64          # Badness score for this design
```

**Example Usage**:
```julia
# Load best density field
density_data = load("output/adversarial_YYYYMMDD_HHMMSS/best_density_field.jld2")
X_best = density_data["X"]
println("Best design from iteration: ", density_data["iteration"])
println("Badness score: ", density_data["badness"])

# Visualize (assuming 60×20 mesh)
using Plots
heatmap(reshape(X_best, (60, 20))', 
        yflip=true, c=:grays, clim=(0,1),
        title="Best Adversarial Design (It $(density_data["iteration"]))", 
        xlabel="X", ylabel="Y")

# Check intermediate density distribution
intermediates = X_best[(X_best .>= 0.1) .& (X_best .<= 0.9)]
println("Intermediate elements: ", length(intermediates), " (", 
        round(100*length(intermediates)/length(X_best), digits=1), "%)")
histogram(X_best, bins=50, 
          xlabel="Density", ylabel="Count", 
          title="Density Distribution (Best Design)")
```

**Size**: ~10 KB (1200 elements × 8 bytes + metadata)

**Why Saved**: Density field `X` is the TopOpt output and cannot be reconstructed from KL coefficients alone (requires re-running 1-2 minute optimization). Saving the best design enables instant visualization and analysis.

---

### `optimization_history.jld2` (Binary Complete History)

**Format**: JLD2 (Julia binary format)

**How to Load**:
```julia
using JLD2
data = load("output/adversarial_YYYYMMDD_HHMMSS/optimization_history.jld2")
```

**Contents** (ESSENTIALS ONLY):
```julia
Dict with keys:
  "history"          => Dict with iteration objectives (metrics only)
  "best_coeffs"      => Matrix{Float64} (best coefficients)
  "final_coeffs"     => Matrix{Float64} (final iteration coefficients)
  "best_badness"     => Float64
  "best_iteration"   => Int
  "final_badness"    => Float64
  "n_modes"          => Dict{Symbol, Int}
  "properties"       => Tuple of Symbols
  "settings"         => Dict with optimization settings
```

**History Dictionary** (objectives only, NO coefficients):
```julia
data["history"] = Dict(
    "iteration"          => [1, 2, 3, ..., 50],           # Int[]
    "badness"            => [0.341, 0.356, ..., 0.478],   # Float64[]
    "compliance"         => [2.89, 3.01, ..., 3.45],      # Float64[]
    "intermediate_frac"  => [0.194, 0.208, ..., 0.256],   # Float64[]
    "severity"           => [0.287, 0.295, ..., 0.320],   # Float64[]
    "gray_indicator"     => [0.169, 0.178, ..., 0.210],   # Float64[]
    "timestamp"          => ["2025-12-01...", ...]        # String[]
)
# NOTE: No coefficient matrices stored in history!
```

**Settings Dictionary**:
```julia
data["settings"] = Dict(
    "max_iterations"  => 50,
    "population_size" => 14,
    "initial_sigma"   => 0.5,
    "weights"         => (0.4, 0.4, 0.2),  # (w_frac, w_severity, w_gray)
    "bounds"          => (-3.0, 3.0)       # Coefficient bounds
)
```

---

### `convergence.png` (Convergence Plots)

**Format**: PNG image (1200×800 pixels)

**Layout**: 4-panel plot

**Panels**:
1. **Top-Left**: Badness vs. Iteration
   - Y-axis: Badness score
   - Shows progress toward maximizing objective

2. **Top-Right**: Compliance vs. Iteration
   - Y-axis: Compliance (log scale)
   - Shows structural performance (not directly optimized)

3. **Bottom-Left**: Intermediate Fraction vs. Iteration
   - Y-axis: Percentage (0-100%)
   - Shows % of elements in gray region

4. **Bottom-Right**: Gray Indicator vs. Iteration
   - Y-axis: GI metric (0-1)
   - Standard TopOpt quality metric

---

## What Gets Saved vs. What Doesn't

### ✅ **Saved**:
- All objective metrics per iteration (badness, compliance, fractions, severity, gray)
- Timestamps per iteration
- Best coefficients (1 matrix - from best iteration)
- Final coefficients (1 matrix - from final iteration)
- **Best density field X (1 vector - from best iteration)**
- Settings and configuration

### ❌ **NOT Saved** (to minimize storage):
- ~~Coefficient matrices from every iteration~~
- ~~Individual design density vectors `X` from every iteration~~ (only best saved!)
- ~~Material field realizations~~
- ~~Element stiffness matrices~~
- ~~VTK files for intermediate iterations~~
- ~~Displacement fields~~
- ~~Checkpoint files~~

**Why**: Only objectives and best/final results matter for analysis. Best density field saved for instant visualization without re-running TopOpt. All intermediate iteration density fields discarded to save space (~50 MB → ~10 KB).

---

## Interpreting Results

### Good Optimization Run

**Characteristics**:
```
Iteration 1:  Badness = 0.342
Iteration 10: Badness = 0.398
Iteration 20: Badness = 0.445
Iteration 30: Badness = 0.467
Iteration 40: Badness = 0.478
Iteration 50: Badness = 0.482  ← Converging, plateauing
```
- ✅ Badness increases steadily
- ✅ Convergence after 30-40 iterations
- ✅ Final intermediate fraction > 30%

### Poor Optimization Run

**Characteristics**:
```
Iteration 1:  Badness = 0.087
Iteration 10: Badness = 0.091
Iteration 20: Badness = 0.089
Iteration 30: Badness = 0.092  ← Stagnant
```
- ❌ Badness stays low
- ❌ No clear improvement trend
- ❌ Final intermediate fraction < 15%

**Fixes**: Adjust weights, increase exploration (initial_sigma), check baseline

---

## Example: Loading and Analyzing Results

```julia
using JLD2, Plots

# Load results
data = load("output/adversarial_20251201_140000/optimization_history.jld2")

# Extract best iteration
best_idx = argmax(data["history"]["badness"])
println("Best iteration: ", data["history"]["iteration"][best_idx])
println("Best badness: ", data["history"]["badness"][best_idx])
println("Best intermediate frac: ", data["history"]["intermediate_frac"][best_idx] * 100, "%")

# Plot convergence
plot(data["history"]["iteration"], data["history"]["badness"],
     xlabel="Iteration", ylabel="Badness", 
     title="Optimization Progress", lw=2, legend=false)

# Get best coefficients
best_coeffs = data["best_coeffs"]
println("Coefficient matrix size: ", size(best_coeffs))

# Analyze which modes are most important
for (j, prop) in enumerate(data["properties"])
    coeffs = best_coeffs[:, j]
    active = coeffs[coeffs .!= 0]
    println("\n$prop:")
    println("  Active modes: ", length(active))
    println("  Max coeff: ", maximum(abs.(active)))
    println("  L2 norm: ", norm(active, 2))
end
```

---

## Summary Table

| Metric | Units | Range | Goal | Weight in Badness |
|--------|-------|-------|------|-------------------|
| Badness | - | 0-1 | MAX | N/A (is objective) |
| Compliance | - | 0-∞ | - | Not in objective |
| Intermediate Frac | fraction | 0-1 | MAX | 40% |
| Severity | - | 0-0.5 | MAX | 40% (normalized) |
| Gray Indicator | - | 0-1 | MAX | 20% |

**Note**: Compliance is recorded but not part of adversarial objective. It shows structural performance but is not directly optimized by outer loop.

---

## File Size Estimates

For 50 iterations with 40 parameters (10 modes × 4 properties), 60×20 mesh (1200 elements):

| File | Typical Size | Purpose |
|------|--------------|---------|
| `optimization_log.txt` | ~5 KB | Iteration objectives (text table) |
| `best_coefficients.txt` | ~10 KB | Best coefficients (human-readable) |
| `final_coefficients.txt` | ~10 KB | Final coefficients (human-readable) |
| `best_density_field.jld2` | ~10 KB | Best density field X (1200 elements) |
| `optimization_history.jld2` | ~10-20 KB | All objectives + best/final coeffs |
| `convergence.png` | ~100-200 KB | Convergence plots |
| **Total** | **~145-255 KB** | Ultra-minimal! |

**Comparison**:
- With all iteration coefficients: ~5-10 MB
- With all iteration density fields: ~50 MB
- With objectives only + best X: **~250 KB** (95-99% reduction!)

**Note**: Size is dominated by convergence plot. Density field adds only ~10 KB while enabling instant visualization.
