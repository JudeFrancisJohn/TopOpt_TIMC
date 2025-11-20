# KL Expansion Seed Usage Guide

## Overview
The `KL_realization` function now supports reproducible random field generation through the `seed` parameter.

## How It Works
The seed controls the random coefficients sampled for the KL expansion modes:
```julia
coeffs = randn(n_modes_final)  # This is now seeded
```

## Usage Examples

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
