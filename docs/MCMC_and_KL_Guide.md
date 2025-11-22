# MCMC and KL Expansion Guide

## Overview
This document explains the Karhunen-Loève (KL) expansion implementation and its use in MCMC (Metropolis-Hastings) for exploring material parameter spaces that lead to "bad" topology-optimized designs.

---

## KL Expansion Theory

The KL expansion represents a spatially-varying random field as:

$$\theta(x) = \mu + \sum_{i=1}^{N} \sqrt{\lambda_i} \xi_i \phi_i(x)$$

Where:
- **μ** = mean value (deterministic - from `MaterialParams`)
- **λᵢ** = eigenvalues from covariance kernel (deterministic)
- **φᵢ(x)** = eigenfunctions/eigenmodes (deterministic)
- **ξᵢ** = random coefficients ~ N(0,1) (stochastic - what MCMC explores)

### Key Insight ⚠️
**Only the coefficients ξᵢ are random!** 

The eigenmodes (λᵢ, φᵢ) are **deterministic** and depend only on:
- Mean material values (e.g., mp.μ_l, mp.μ_t)
- Covariance structure (σ, correlation length Lc, kernel type)
- Spatial geometry (mesh coordinates)

This separation is **crucial** for efficient MCMC implementation.

---

## Implementation Architecture

### Three-Function Design

The refactored implementation separates deterministic and stochastic parts:

#### 1. `compute_KL_eigenmodes()` - Solve eigenvalue problem ONCE
```julia
kl_modes = compute_KL_eigenmodes(mp, coords_elem, :μ_l, sigma;
    Lc=0.01, N_modes=80, kernel=:exponential, mode=:lognormal, ...)
```
- **Purpose**: Compute deterministic eigenmodes λᵢ and φᵢ
- **When to call**: Once before any sampling begins
- **Returns**: `KL_Eigenmodes` struct containing eigenmodes
- **Cost**: Expensive (eigenvalue decomposition)

#### 2. `sample_KL_field()` - Generate realizations efficiently
```julia
field = sample_KL_field(kl_modes, coeffs; eltype_out=Float32)
```
- **Purpose**: Generate material field from eigenmodes + coefficients
- **When to call**: Many times with different coefficients
- **Returns**: Sampled field (vector or array)
- **Cost**: Cheap (matrix-vector multiplication)

#### 3. `KL_realization()` - Legacy wrapper (backward compatible)
```julia
fields = KL_realization(mp, coords_elem; 
    σs=Dict(:μ_l=>0.1*mp.μ_l), provided_coeffs=coeffs_dict, seed=42)
```
- **Purpose**: Combines steps 1 and 2 in one call
- **When to use**: Simple one-off sampling, or for backward compatibility
- **Cost**: Expensive (recomputes eigenmodes each time)
- **Note**: Inefficient for MCMC - use separate functions instead!

---

## MCMC Workflow (Correct Approach)

### ✅ Efficient Implementation

```julia
# STEP 1: Pre-compute eigenmodes ONCE (before MCMC loop)
println("Pre-computing eigenmodes...")
kl_μ_l = compute_KL_eigenmodes(mp, coords_elem, :μ_l, 0.1*mp.μ_l; 
                               Lc=0.01, N_modes=80, kernel=:exponential, mode=:lognormal)
kl_μ_t = compute_KL_eigenmodes(mp, coords_elem, :μ_t, 0.1*mp.μ_t; ...)
kl_α = compute_KL_eigenmodes(mp, coords_elem, :α, 0.1*mp.alpha; ...)
kl_β = compute_KL_eigenmodes(mp, coords_elem, :β, 0.1*mp.beta; ...)

# STEP 2: MCMC loop - only coefficients change
for iteration in 1:N_MCMC
    # Propose new coefficients (random walk)
    proposal_coeffs[:μ_l] = current_coeffs[:μ_l] + σ_step * randn(80)
    
    # Generate fields using FIXED eigenmodes
    field_μ_l = sample_KL_field(kl_μ_l, proposal_coeffs[:μ_l])
    field_μ_t = sample_KL_field(kl_μ_t, proposal_coeffs[:μ_t])
    # ... etc
    
    # Run topology optimization with these fields
    # Evaluate badness score
    # Metropolis-Hastings accept/reject
end
```

### ❌ Inefficient (Old) Implementation

```julia
# WRONG: Recomputes eigenmodes every iteration!
for iteration in 1:N_MCMC
    # This recomputes eigenvalue decomposition each time (very slow!)
    fields = KL_realization(mp, coords_elem; 
        provided_coeffs=proposal_coeffs, ...)
    
    # This wastes ~90% of computation time on re-solving eigenvalue problems
end
```

**Why is this wrong?**
- Eigenmodes are **deterministic** - they shouldn't change between iterations
- MCMC should explore **coefficient space**, not re-solve eigenvalue problems
- Wastes computation on redundant eigenvalue decompositions

---

## Main MCMC Driver: `test/MC_run_v2.jl`

**This is the official MCMC driver script.**

### Purpose
Explore material parameter fields that lead to topology-optimized designs with high intermediary densities (0.3-0.7), which are considered "bad" designs in SIMP-based topology optimization.

### Research Question
*Which spatially-varying material parameter fields cause the optimizer to get stuck with gray/smudged designs instead of clean binary (0/1) topologies?*

### How It Works

#### Phase 1: Pre-computation (Once at Startup)
```julia
# Compute eigenmodes for all 4 material properties
kl_modes_dict = Dict{Symbol, KL_Eigenmodes}()
for (prop_sym, sigma) in KL_SIGMA_DICT
    kl_modes_dict[prop_sym] = compute_KL_eigenmodes(mp, coords_elem, prop_sym, sigma; 
        Lc=KL_LC, N_modes=N_MODES, kernel=KL_KERNEL, mode=KL_MODE, ...)
end
```

#### Phase 2: MCMC Sampling (Main Loop)
```julia
for iteration in 2:N_CHAIN
    # 1. Propose new coefficients (Random Walk Metropolis)
    proposal_coeffs[:μ_l] = current_coeffs[:μ_l] + PROPOSAL_SIGMA * randn(N_MODES)
    
    # 2. Generate fields using FIXED eigenmodes
    fields = run_with_kl_coeffs(iteration, proposal_coeffs, kl_modes_dict)
    
    # 3. Run topology optimization
    # 4. Evaluate badness metric
    badness = DENSITY_WEIGHT * (intermediary_density_proportion) 
            + COMPLIANCE_WEIGHT * compliance
    
    # 5. Metropolis-Hastings acceptance
    acceptance_prob = exp(BETA * (proposal_badness - current_badness))
    if rand() < acceptance_prob
        accept_proposal()
    end
end
```

### Configuration Parameters

```julia
# MCMC Settings
const N_CHAIN = 20              # Total MCMC iterations
const BURN_IN = 6               # Burn-in period (30% of chain)
const PROPOSAL_SIGMA = 0.5      # Random walk step size for coefficients
const BETA = 50.0               # Inverse temperature (higher = prefer worse designs)

# Badness Metric Weights
const DENSITY_WEIGHT = 5.0      # Weight for intermediary density proportion
const COMPLIANCE_WEIGHT = 1e-3  # Weight for compliance (small)

# KL Expansion Configuration (MUST BE CONSISTENT!)
const N_MODES = 80              # Number of KL modes per property
const KL_SIGMA_DICT = Dict(
    :μ_l => 0.1 * mp.μ_l,       # 10% coefficient of variation
    :μ_t => 0.1 * mp.μ_t,
    :α   => 0.1 * mp.alpha,
    :β   => 0.1 * mp.beta
)
const KL_LC = 0.01              # Correlation length
const KL_KERNEL = :exponential  # Covariance kernel type
const KL_MODE = :lognormal      # Transformation mode
const KL_USE_CENTROIDS = false  # Use per-node fields
const KL_MAKE_SPARSE = true     # Sparse covariance matrix
```

### Badness Metric

The "badness" score quantifies how undesirable a design is:

```julia
function evaluate_badness(log_entry)
    # Extract density distribution from final iteration
    bin_proportions = log_entry.final_density_log.bin_proportions
    
    # Intermediary density proportion (bin 2: between 0.3 and 0.7)
    mid_prop = bin_proportions[2]
    
    # Total badness
    return DENSITY_WEIGHT * mid_prop + COMPLIANCE_WEIGHT * compliance
end
```

Higher scores are "worse" designs (more intermediary densities), which is what we want MCMC to find!

### Output

Results saved to `output/mcmc_chain_YYYYMMDD_HHMMSS.jld2`:

```julia
# Saved variables:
- chain_records      # Full history: coeffs, scores, log_entries per iteration
- chain_scores       # Badness scores over all iterations
- post_burnin_scores # Scores after burn-in (for analysis)
- accepted_count     # Number of accepted proposals
- All configuration parameters (for reproducibility)
```

---

## Monte Carlo Sampling (Non-MCMC)

For **independent** Monte Carlo samples (not MCMC chains), use:

```julia
# In COPY_stochastic_modified_v2_MC.jl
run_logs = multiple_runs(100)  # 100 independent TopOpt runs
```

This function:
- Uses different random seeds for each run
- Generates truly independent material field samples
- Each sample uses `KL_realization()` with a unique seed
- Useful for statistical analysis of design variability

---

## Key Differences: MCMC vs Monte Carlo

| Aspect | MCMC (MC_run_v2.jl) | Monte Carlo (multiple_runs) |
|--------|---------------------|------------------------------|
| **Purpose** | Find bad-design-inducing materials | Statistical sampling |
| **Sampling** | Correlated (Markov chain) | Independent |
| **Efficiency** | Pre-compute eigenmodes once | Recompute each time (OK for small N) |
| **Coefficients** | Explored via random walk | Random each iteration |
| **Seeds** | Not used (coefficients evolve) | Different seed per run |
| **Output** | Chain exploring parameter space | Independent samples |

---

## Reproducing MCMC Results

To reproduce an MCMC chain exactly:

1. Load the saved `.jld2` file
2. Extract configuration parameters:
   ```julia
   using JLD2
   @load "mcmc_chain_20251122_150000.jld2"
   
   # Configuration is saved in the file
   println(KL_SIGMA_DICT)
   println(PROPOSAL_SIGMA)
   println(BETA)
   ```

3. Re-run with same configuration:
   ```julia
   # Edit MC_run_v2.jl to use saved parameters
   # Pre-compute eigenmodes with SAME parameters
   # Results will be deterministic if coefficients are replayed
   ```

---

## Best Practices

### For MCMC
✅ **DO:**
- Pre-compute eigenmodes before the MCMC loop
- Use `sample_KL_field()` inside the loop
- Keep KL parameters (σ, Lc, kernel) consistent
- Save all configuration parameters with results
- Monitor acceptance rate (target ~20-40%)

❌ **DON'T:**
- Call `KL_realization()` inside MCMC loop (inefficient)
- Change KL parameters mid-chain
- Use seeds in MCMC (coefficients are being explored)

### For Monte Carlo
✅ **DO:**
- Use different seeds for each run
- Document seed values
- Use `KL_realization()` for simplicity (OK for small N)

### For Research/Publications
✅ **DO:**
- Document all KL parameters (σ, Lc, N_modes, kernel)
- Save configuration with results
- Report acceptance rates for MCMC
- Show burn-in period clearly

---

## Troubleshooting

### "Eigenmodes change between iterations"
❌ You're using `KL_realization()` inside MCMC loop  
✅ Pre-compute with `compute_KL_eigenmodes()` once

### "MCMC acceptance rate is 0%"
- PROPOSAL_SIGMA is too large
- Try reducing from 0.5 to 0.1 or 0.05

### "MCMC acceptance rate is 100%"
- PROPOSAL_SIGMA is too small
- BETA might be too low
- Try increasing PROPOSAL_SIGMA

### "Results not reproducible"
- For MCMC: This is expected (random walk)
- For MC: Check that you're using the same seeds

---

## References

### Code Files
- **MCMC Driver**: `test/MC_run_v2.jl` (main script)
- **KL Implementation**: `utils/stochastic_utils.jl`
- **TopOpt Driver**: `src/COPY_stochastic_modified_v2_MC.jl`
- **MCMC Utilities**: `utils/mcmc_utils.jl`

### Related Documentation
- `docs/KL_seed_usage.md` - Legacy seed documentation (pre-refactor)
- `.github/copilot-instructions.md` - Project overview

---

**Last Updated**: November 22, 2025  
**Version**: 2.0 (post-refactor)
