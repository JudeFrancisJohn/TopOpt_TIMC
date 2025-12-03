# Adversarial Topology Optimization Guide

## Overview

This guide explains the adversarial optimization framework for finding KL expansion coefficients that maximize intermediate densities in SIMP topology optimization.

## Conceptual Framework

### The Nested Optimization Problem

1. **Inner Loop (SIMP Topology Optimization)**:
   - Goal: Minimize compliance subject to volume constraint
   - Penalty: Pushes densities toward 0 (void) or 1 (solid)
   - Input: Material parameter field (spatially heterogeneous)

2. **Outer Loop (Adversarial Coefficient Search)**:
   - Goal: Find material heterogeneity patterns that confuse the optimizer
   - Mechanism: Search for KL coefficients that create "difficult" material distributions
   - Output: Coefficients that produce designs with many intermediate densities

### Why This Matters

Intermediate densities (gray regions) in SIMP optimization are:
- **Theoretically problematic**: Represent ambiguous material states
- **Computationally expensive**: Slow convergence
- **Physically unrealistic**: Cannot be manufactured

Finding coefficients that induce intermediate densities helps us:
1. Understand which material heterogeneities cause optimization difficulties
2. Design more robust optimization algorithms
3. Develop better material filtering strategies

## Architecture

### Key Components

#### 1. KL Expansion Infrastructure (`utils/stochastic_utils.jl`)

**Pre-computation (deterministic, done once)**:
```julia
kl_modes = compute_KL_eigenmodes(
    material_params, coords_elem, prop_sym, sigma;
    Lc=0.01, N_modes=15, use_centroids=false, make_sparse=false
)
```
- Solves eigenvalue problem for covariance kernel
- Returns `KL_Eigenmodes` struct with eigenvalues and eigenvectors
- **Critical**: Eigenmodes do NOT change during optimization

**Field generation (stochastic, done each iteration)**:
```julia
field = sample_KL_field(
    kl_modes,  # Pre-computed eigenmodes (constant)
    coeffs;    # Variable coefficients (updated by optimizer)
    eltype_out=Float32
)
```
- Generates material field from eigenmodes + coefficients
- Efficient: No eigenvalue solve needed
- Different coefficients → different material realizations

#### 2. Adversarial Metrics (`utils/adversarial_utils.jl`)

**Intermediate Density Fraction**:
```julia
frac = compute_intermediary_fraction(X; threshold_low=0.1, threshold_high=0.9)
```
- Percentage of elements in intermediate range [0.1, 0.9]
- Simple but effective metric

**Intermediate Density Severity**:
```julia
severity = compute_intermediary_severity(X; threshold_low=0.1, threshold_high=0.9)
```
- Weights intermediate densities by proximity to 0.5
- Elements near 0.5 contribute more than elements near boundaries

**Gray Indicator**:
```julia
gray = compute_gray_indicator(X)
```
- Standard topology optimization metric: `GI = (4/n) * sum(x_i * (1 - x_i))`
- Range: 0 (binary) to 1 (all at 0.5)

**Combined Badness**:
```julia
badness = compute_combined_badness(X, c; w_frac=0.4, w_sev=0.4, w_gray=0.2)
```
- Weighted combination of all metrics
- Higher = more intermediate densities = "worse" for SIMP

#### 3. Optimization Strategy (`utils/adversarial_optimizer.jl`)

**Optimizer Configuration**:
```julia
opt = AdversarialOptimizer(
    n_modes, properties;
    max_iterations=50,
    population_size=0,  # Auto-sized
    initial_sigma=0.5,
    coeff_bounds=(-3.0, 3.0),
    w_frac=0.4, w_severity=0.4, w_gray=0.2
)
```

**Why Adaptive Differential Evolution?**
- **Gradient-free**: No need for sensitivity analysis through TopOpt
- **Robust**: Handles noisy objectives (TopOpt convergence varies)
- **High-dimensional**: Works with 60 parameters (15 modes × 4 properties)
- **Bounded**: Respects coefficient bounds naturally

#### 4. Main Execution (`test/proxy.jl`)

**Objective Function**:
```julia
function evaluate_objective(coeffs_mat)
    # 1. Convert coefficients to Dict
    coeffs_dict = matrix_to_coeffs_dict(coeffs_mat, PROPERTIES, n_modes)
    
    # 2. Generate material fields (using fixed eigenmodes!)
    for prop_sym in PROPERTIES
        result_fields[prop_sym] = sample_KL_field(
            kl_modes_dict[prop_sym],  # CONSTANT
            coeffs_dict[prop_sym]     # VARIABLE
        )
    end
    
    # 3. Build material field and validate
    mf = build_material_field(result_fields; ...)
    validate_material_field(mf, material_params)
    
    # 4. Run topology optimization
    build_KEStore!(dh, mf, nnodes_loc, avg_mp_store)
    X, c = topopt_run(1)
    
    # 5. Compute adversarial metrics
    badness = compute_combined_badness(X, c)
    
    return badness, c, X, diagnostics
end
```

## Usage

### Running the Optimization

```julia
# From Julia REPL
include("test/proxy.jl")
main()

# Or directly from command line
julia --project=. test/proxy.jl
```

### Configuration Parameters

Edit `test/proxy.jl` to adjust:

```julia
# Number of KL modes per property (dimensionality tradeoff)
const N_MODES_PER_PROP = 15  # Start small (15), increase if needed

# Optimization settings
max_iterations = 50      # More iterations = better solution, slower
population_size = 0      # 0 = auto-sized (recommended)

# Objective weights
w_frac = 0.4        # Intermediate fraction weight
w_severity = 0.4    # Severity weight
w_gray = 0.2        # Gray indicator weight

# Coefficient bounds
coeff_bounds = (-3.0, 3.0)  # Restrict to reasonable range
```

### Output Structure

```
output/adversarial_YYYYMMDD_HHMMSS/
├── optimization_log.txt           # Iteration-by-iteration progress
├── best_coefficients.txt          # Best coefficients found
├── final_coefficients.txt         # Final iteration coefficients
├── optimization_history.jld2      # Complete history (JLD2 format)
├── convergence.png                # Convergence plots
└── checkpoints/
    ├── coeffs_iter_10.txt
    ├── history_iter_10.jld2
    ├── coeffs_iter_20.txt
    └── ...
```

### Interpreting Results

**Best Coefficients File**:
```
# KL Expansion Coefficients
# Properties: μ_l, μ_t, α, β

## STATISTICS
## μ_l
  max_abs     : 2.345678e+00
  mean        : -1.234567e-02
  n_active    : 15
  norm_l2     : 3.456789e+00
  norm_linf   : 2.345678e+00
  std         : 8.901234e-01

## COEFFICIENTS
## μ_l (15 modes)
   1      μ_l      1.2345678901e+00
   2      μ_l     -8.7654321098e-01
   ...
```

**Key Metrics**:
- `norm_l2`: Total energy in coefficients (lower = smoother fields)
- `max_abs`: Largest coefficient magnitude (check for extremes)
- `n_active`: Number of non-zero modes used

## Reconstructing Material Fields from Coefficients

Once you have optimized coefficients, reconstruct the material field:

```julia
using JLD2

# Load results
data = load("output/adversarial_YYYYMMDD_HHMMSS/optimization_history.jld2")
best_coeffs = data["best_coeffs"]
n_modes = data["n_modes"]
properties = data["properties"]

# Pre-compute eigenmodes (same as optimization)
kl_modes_dict = Dict{Symbol, Any}()
for prop_sym in properties
    kl_modes_dict[prop_sym] = compute_KL_eigenmodes(
        material_params, coords_elem, prop_sym, σs[prop_sym];
        Lc=0.01, N_modes=n_modes[prop_sym], 
        use_centroids=false, make_sparse=false
    )
end

# Convert coefficients to Dict
coeffs_dict = matrix_to_coeffs_dict(best_coeffs, properties, n_modes)

# Generate material fields
result_fields = Dict{Symbol, Any}()
for prop_sym in properties
    result_fields[prop_sym] = sample_KL_field(
        kl_modes_dict[prop_sym], 
        coeffs_dict[prop_sym]; 
        eltype_out=Float32
    )
end

# Add constant properties
result_fields[:λ] = fill(Float32(material_params.λ), N_ELEM, N_LOC)
result_fields[:angle] = fill(Float32(material_params.angle), N_ELEM, N_LOC)

# Build material field
mf = build_material_field(result_fields; use_centroids=false, eltype_out=Float32)

# Run topology optimization
build_KEStore!(dh, mf, nnodes_loc, avg_mp_store)
X, c = topopt_run(1)
```

## Troubleshooting

### Issue: Optimization converges to invalid material fields

**Symptom**: Warnings about negative μ_l or μ_t values

**Solution**:
1. Tighten coefficient bounds: `coeff_bounds = (-2.0, 2.0)`
2. Reduce standard deviations in `σs` dictionary
3. Increase `tolerance_factor` in `validate_material_field`

### Issue: Optimization is too slow

**Symptom**: Each iteration takes > 5 minutes

**Solution**:
1. Reduce `N_MODES_PER_PROP` (try 10 or 8)
2. Reduce `maxiter` in `topopt_run` (edit driver file)
3. Reduce mesh resolution (`nelx`, `nely` in `input/params_geom.jl`)

### Issue: No improvement in badness metric

**Symptom**: Badness stays near zero or decreases

**Solution**:
1. Check that inner TopOpt is working correctly (run deterministic case first)
2. Adjust objective weights (try `w_frac=0.6, w_sev=0.3, w_gray=0.1`)
3. Increase `initial_sigma` for more exploration
4. Check that eigenmodes are non-trivial (`print(kl_modes.eigenvalues)`)

### Issue: Eigenmodes change between runs

**Symptom**: Different results with same coefficients

**Solution**:
- **This should NOT happen!** Eigenmodes are deterministic.
- If it does, check:
  1. `make_sparse` setting is consistent
  2. Mesh (`coords_elem`) hasn't changed
  3. Material parameters are identical
  4. No random seed affects eigenmode computation

## Advanced Topics

### Sensitivity Analysis of Modes

Which KL modes contribute most to intermediate densities?

```julia
# After optimization, analyze coefficient magnitudes
for prop in properties
    coeffs = best_coeffs[:, findfirst(==(prop), properties)]
    sorted_idx = sortperm(abs.(coeffs), rev=true)
    println("Top 5 modes for $prop:")
    for i in 1:5
        @printf("  Mode %2d: %.4f (eigenvalue: %.4e)\n", 
                sorted_idx[i], coeffs[sorted_idx[i]], 
                kl_modes_dict[prop].eigenvalues[sorted_idx[i]])
    end
end
```

### Multi-Start Optimization

Run multiple optimizations with different initializations:

```julia
function multi_start_optimization(n_starts=5)
    best_overall = -Inf
    best_coeffs_overall = nothing
    
    for i in 1:n_starts
        println("\n" * "="^80)
        println("Multi-Start Run $i / $n_starts")
        println("="^80)
        
        # Use different seed each time
        Random.seed!(42 + i)
        
        coeffs, badness, result = run_adversarial_optimization(
            max_iterations=30,  # Shorter runs
            population_size=0
        )
        
        if badness > best_overall
            best_overall = badness
            best_coeffs_overall = coeffs
        end
    end
    
    return best_coeffs_overall, best_overall
end
```

### Dimension Reduction via PCA

For very high-dimensional problems, consider optimizing in PCA space:

```julia
# Generate sample coefficient sets
n_samples = 100
sample_coeffs = [randn(max_modes, n_props) for _ in 1:n_samples]

# Evaluate and collect
X_data = hcat([vec(c) for c in sample_coeffs]...)  # Each column is a sample

# PCA
using MultivariateStats
M = fit(PCA, X_data; maxoutdim=20)  # Reduce to 20 principal components

# Optimize in PCA space
function objective_pca(z::Vector{Float64})
    # Transform from PCA space to coefficient space
    x_full = reconstruct(M, z)
    coeffs_mat = reshape(x_full, max_modes, n_props)
    
    badness, _, _, _ = evaluate_objective(coeffs_mat)
    return -badness
end
```

## References

- **SIMP Method**: Bendsøe & Sigmund (2003), "Topology Optimization"
- **KL Expansion**: Ghanem & Spanos (1991), "Stochastic Finite Elements"
- **Gray Scale Filtering**: Sigmund (2007), "Morphology-based black and white filters"
- **Adversarial Optimization**: Goodfellow et al. (2014), "Generative Adversarial Networks" (conceptual inspiration)

## Contact

For questions or issues, refer to the main project documentation in `README.md`.
