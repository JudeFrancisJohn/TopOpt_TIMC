# KL Expansion and MCMC Usage Guide

## Overview
This document explains the KL (Karhunen-Loève) expansion implementation and its use in both Monte Carlo sampling and MCMC (Metropolis-Hastings) algorithms for exploring material parameter spaces.

## KL Expansion Theory

The KL expansion represents a spatially-varying random field as:

$$\theta(x) = \mu + \sum_{i=1}^{N} \sqrt{\lambda_i} \xi_i \phi_i(x)$$

Where:
- **μ** = mean value (deterministic)
- **λᵢ** = eigenvalues from covariance kernel (deterministic)
- **φᵢ(x)** = eigenfunctions/eigenmodes (deterministic)
- **ξᵢ** = random coefficients ~ N(0,1) (stochastic)

### Key Insight
**Only the coefficients ξᵢ are random!** The eigenmodes are deterministic and depend only on:
- Mean material values
- Covariance structure (σ, correlation length Lc, kernel type)
- Spatial geometry (mesh coordinates)

## New Implementation (v2)

### Three-Function Architecture

The implementation now separates the deterministic and stochastic parts:

1. **`compute_KL_eigenmodes()`** - Solve eigenvalue problem ONCE (deterministic)
   - Computes λᵢ and φᵢ based on covariance structure
   - Returns `KL_Eigenmodes` struct containing eigenmodes
   - Should be called once before sampling begins

2. **`sample_KL_field()`** - Generate realizations from eigenmodes (stochastic)
   - Takes pre-computed eigenmodes + coefficient vector
   - Efficiently generates material field: μ + Σ√(λᵢ)ξᵢφᵢ
   - Called many times with different coefficients

3. **`KL_realization()`** - Legacy wrapper (less efficient)
   - Combines steps 1 and 2 in one call
   - Recomputes eigenmodes each time (inefficient for multiple samples)
   - Maintained for backward compatibility

### Why This Matters for MCMC

In MCMC, you want to explore the **coefficient space** (ξᵢ), not re-solve eigenvalue problems:

```julia
# WRONG (old approach - recomputes eigenmodes each iteration):
for mcmc_iteration in 1:1000
    fields = KL_realization(mp, coords_elem; provided_coeffs=coeffs)  # Slow!
end

# CORRECT (new approach - eigenmodes computed once):
# Step 1: Pre-compute eigenmodes (once)
kl_modes = compute_KL_eigenmodes(mp, coords_elem, :μ_l, sigma; ...)

# Step 2: Sample many times efficiently
for mcmc_iteration in 1:1000
    field = sample_KL_field(kl_modes, coeffs)  # Fast!
end
```

## Main MCMC Driver: `MC_run_v2.jl`

**`test/MC_run_v2.jl`** is now the official MCMC driver for exploring material parameter spaces.

### Purpose
Find material parameter fields that lead to "bad" topology-optimized designs (high intermediary densities), even though the optimization algorithm tries to produce binary (0/1) designs.

### How It Works

1. **Pre-computation Phase** (done once at startup)
   ```julia
   # Compute fixed eigenmodes for all material properties
   kl_μ_l = compute_KL_eigenmodes(mp, coords_elem, :μ_l, σ_μ_l; Lc=0.01, N_modes=80, ...)
   kl_μ_t = compute_KL_eigenmodes(mp, coords_elem, :μ_t, σ_μ_t; ...)
   # ... etc for :α, :β
   ```

2. **MCMC Loop** (explores coefficient space)
   ```julia
   for iteration in 1:N_CHAIN
       # Propose new coefficients (random walk)
       proposal_coeffs[:μ_l] = current_coeffs[:μ_l] + PROPOSAL_SIGMA * randn(80)
       
       # Generate fields using FIXED eigenmodes
       field_μ_l = sample_KL_field(kl_μ_l, proposal_coeffs[:μ_l])
       
       # Run topology optimization
       # Evaluate "badness" score
       # Accept/reject via Metropolis-Hastings
   end
   ```

3. **Badness Metric**
   ```julia
   badness = DENSITY_WEIGHT * (proportion of intermediary densities) 
           + COMPLIANCE_WEIGHT * compliance
   ```

### Configuration Parameters

```julia
const N_CHAIN = 20              # MCMC iterations
const BURN_IN = 6               # Discard first 30% as burn-in
const N_MODES = 80              # Number of KL modes per property
const PROPOSAL_SIGMA = 0.5      # Random walk step size
const BETA = 50.0               # Inverse temperature (higher = prefer bad designs)
const DENSITY_WEIGHT = 5.0      # Weight for intermediary density proportion
const COMPLIANCE_WEIGHT = 1e-3  # Weight for compliance (small)

# KL Expansion parameters
const KL_SIGMA_DICT = Dict(
    :μ_l => 0.1 * mp.μ_l,      # 10% COV for longitudinal shear
    :μ_t => 0.1 * mp.μ_t,      # 10% COV for transverse shear
    :α   => 0.1 * mp.alpha,    # 10% COV for alpha
    :β   => 0.1 * mp.beta      # 10% COV for beta
)
const KL_LC = 0.01              # Correlation length
const KL_KERNEL = :exponential  # Covariance kernel
const KL_MODE = :lognormal      # Transformation mode
```

### Output

Results saved to `output/mcmc_chain_YYYYMMDD_HHMMSS.jld2` containing:
- `chain_records`: Full history of all iterations
- `chain_scores`: Badness scores over time
- `post_burnin_scores`: Scores after burn-in period
- All configuration parameters for reproducibility

## Legacy: `Simple MC_run.jl`

**Deprecated.** Use `MC_run_v2.jl` instead.

The old file still uses `run_single_design()` which calls `KL_realization()` internally, recomputing eigenmodes every iteration. This is inefficient and conceptually incorrect for MCMC.

## Monte Carlo Sampling (Multiple Independent Runs)

For standard Monte Carlo (not MCMC), use `multiple_runs()` from the main driver:

```julia
# In COPY_stochastic_modified_v2_MC.jl
run_logs = multiple_runs(100)  # 100 independent samples
```

This uses different random seeds for each run, generating truly independent samples.

## Seed Usage (for non-MCMC runs)

### 1. Single Reproducible Run
```julia
# Always generates the same material field
fields = KL_realization(mp, coords_elem;
    σs = Dict(:μ_l => 0.5 * mp.μ_l, :μ_t => 0.5 * mp.μ_t),
    Lc=0.05, N_modes=80,
    seed=42)  # Fixed seed = reproducible result
```

### 2. Multiple Different Runs (but reproducible)
```julia
for run_id in 1:10
    fields = KL_realization(mp, coords_elem;
        σs = Dict(:μ_l => 0.5 * mp.μ_l, :μ_t => 0.5 * mp.μ_t),
        Lc=0.05, N_modes=80,
        seed=1000 + run_id)  # Seeds: 1001, 1002, ..., 1010
    # Each run is different but repeatable
end
```

### 3. Monte Carlo Simulations
```julia
function monte_carlo_analysis(n_samples=100, base_seed=1000)
    results = []
    for i in 1:n_samples
        fields = KL_realization(mp, coords_elem;
            σs = Dict(:μ_l => 0.5 * mp.μ_l, :μ_t => 0.5 * mp.μ_t),
            Lc=0.05, N_modes=80,
            seed=base_seed + i)
        
        # Run FEA and topology optimization
        # ... your code ...
        
        push!(results, computed_compliance)
    end
    return results
end
```

### 4. Parameter Sweep with Reproducibility
```julia
# Study effect of different σ values with same underlying random field
base_seed = 2024
for sigma_factor in [0.1, 0.3, 0.5, 0.7, 1.0]
    fields = KL_realization(mp, coords_elem;
        σs = Dict(:μ_l => sigma_factor * mp.μ_l),
        Lc=0.05, N_modes=80,
        seed=base_seed)  # Same seed = same random coefficients
    # Only sigma changes, random pattern stays same
end
```

### 5. No Seed (Random Behavior)
```julia
# Different result every time (not recommended for research)
fields = KL_realization(mp, coords_elem;
    σs = Dict(:μ_l => 0.5 * mp.μ_l),
    seed=nothing)  # or omit the seed parameter entirely
```

## Best Practices

### For Single Runs
- Use a memorable seed like `42`, `123`, `2024`, etc.
- Document the seed in your results/output files

### For Multiple Runs
- Use `base_seed + run_index` pattern (e.g., `1000 + i`)
- This ensures all runs are different but reproducible
- Record the base seed in your output

### For Publications/Research
- **Always use seeds** for reproducibility
- Document seeds in your paper/report
- Use different seed ranges for different experiments:
  - Experiment A: seeds 1000-1099
  - Experiment B: seeds 2000-2099
  - etc.

### For Debugging
- Use the same seed during debugging
- Change seed only when you want to verify robustness

## Technical Notes

### What Gets Seeded
- The random normal coefficients for each KL mode
- Each property (μ_l, μ_t, α, β) draws from the same seeded RNG
- The covariance matrix and eigenvectors are deterministic (not random)

### What Doesn't Get Seeded
- The mesh coordinates (deterministic input)
- The correlation length Lc (deterministic parameter)
- The number of modes (deterministic parameter)
- The eigendecomposition (deterministic given covariance matrix)

### Seed Scope
- The seed is set at the **start** of `KL_realization`
- It affects all random draws within that function call
- Multiple properties (μ_l, μ_t, α, β) share the same seeded sequence
- If you call `KL_realization` twice with the same seed, you get identical results

## Example Output Naming
Include seed in your output filenames for traceability:
```julia
seed = 1000 + run_id
fields = KL_realization(mp, coords_elem; ..., seed=seed)
output_file = "result_seed$(seed)_Lc$(Lc)_N$(N_modes).vtu"
```

## Sigma Values (Your Current Setup)
You're passing σs as percentages of mean values:
```julia
σs = Dict(
    :μ_l => 0.5 * mp.μ_l,   # 50% coefficient of variation
    :μ_t => 0.5 * mp.μ_t,   # 50% coefficient of variation
    :α   => 0.5 * mp.alpha, # 50% coefficient of variation
    :β   => 0.5 * mp.beta   # 50% coefficient of variation
)
```
This is good practice! The standard deviation is 50% of the mean value.
