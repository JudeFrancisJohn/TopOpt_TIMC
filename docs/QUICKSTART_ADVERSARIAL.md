# Quick Start Guide: Adversarial Optimization

## What Does This Do?

Finds KL expansion coefficients that make SIMP topology optimization produce designs with lots of intermediate (gray) densities, despite the optimizer's penalty trying to push densities to 0 or 1.

## Why?

Understanding which material heterogeneity patterns confuse the optimizer helps:
1. Design more robust optimization algorithms
2. Identify problematic material distributions
3. Develop better filtering strategies

---

## Installation

### 1. Install Required Package
```julia
# In Julia REPL or from command line
julia --project=. -e 'using Pkg; Pkg.add("BlackBoxOptim")'
```

### 2. Verify Setup
```julia
# Quick test (2-5 minutes)
julia --project=. test/test_adversarial_setup.jl
```

Expected output:
```
[1/6] Loading TopOpt driver...
✓ Modules loaded successfully

[2/6] Testing eigenmode computation...
  μ_l: 10 modes, λ ∈ [1.234e-03, 5.678e-02]
  μ_t: 10 modes, λ ∈ [1.234e-03, 5.678e-02]
  ...
✓ Eigenmodes computed successfully

...

✓ All systems operational - ready for adversarial optimization!
```

---

## Running Optimization

### Basic Usage

```julia
# From command line
julia --project=. test/proxy.jl
```

### Customizing Parameters

Edit `test/proxy.jl` before running:

```julia
# Line ~60: Number of KL modes (dimensionality)
const N_MODES_PER_PROP = 15  # Start with 15, increase if needed

# Line ~336: Optimization settings
max_iterations = 50      # More = better solution, slower
population_size = 0      # 0 = auto (recommended)
```

### Execution Time

- **15 modes, 50 iterations**: ~30-60 minutes
- **10 modes, 30 iterations**: ~15-30 minutes (faster testing)
- **20 modes, 100 iterations**: ~2-3 hours (thorough search)

---

## Understanding Output

### During Optimization

```
================================================================================
Iteration  23
--------------------------------------------------------------------------------
  Badness:              0.342156
  Compliance:           1.234567e+02
  Intermediate Frac:    0.3421 (34.2%)
  Severity:             0.245678
  Gray Indicator:       0.456789
================================================================================
```

**Key Metrics**:
- **Badness**: Combined score (higher = more intermediate densities)
- **Intermediate Frac**: % of elements with density in [0.1, 0.9]
- **Severity**: How close intermediate densities are to 0.5
- **Gray Indicator**: Standard TopOpt metric (0=binary, 1=all gray)

### After Completion

```
output/adversarial_20251129_143022/
├── best_coefficients.txt          ← Load this to reconstruct best field
├── optimization_log.txt           ← Iteration history
├── optimization_history.jld2      ← Full data (for analysis)
└── convergence.png                ← Visual progress
```

---

## Reconstructing Best Material Field

Once optimization completes, reconstruct the "worst" material field:

```julia
using JLD2

# 1. Load results
results = load("output/adversarial_YYYYMMDD_HHMMSS/optimization_history.jld2")
best_coeffs = results["best_coeffs"]
n_modes = results["n_modes"]
properties = results["properties"]

# 2. Load TopOpt infrastructure
include("src/COPY_stochastic_modified_v2_MC.jl")
include("input/params_MCMC.jl")

# 3. Re-compute eigenmodes (same as optimization)
σs = Dict(:μ_l => 0.8, :μ_t => 0.8, :α => 0.8, :β => 0.8)
kl_modes_dict = Dict{Symbol, Any}()

for prop_sym in properties
    kl_modes_dict[prop_sym] = compute_KL_eigenmodes(
        material_params, coords_elem, prop_sym, σs[prop_sym];
        Lc=0.01, N_modes=n_modes[prop_sym], 
        use_centroids=false, make_sparse=false
    )
end

# 4. Generate fields from best coefficients
include("utils/adversarial_utils.jl")
coeffs_dict = matrix_to_coeffs_dict(best_coeffs, properties, n_modes)

result_fields = Dict{Symbol, Any}()
for prop_sym in properties
    result_fields[prop_sym] = sample_KL_field(
        kl_modes_dict[prop_sym], 
        coeffs_dict[prop_sym]; 
        eltype_out=Float32
    )
end

# Add constants
n_elem, n_loc = size(coords_elem)
result_fields[:λ] = fill(Float32(material_params.λ), n_elem, n_loc)
result_fields[:angle] = fill(Float32(material_params.angle), n_elem, n_loc)

# 5. Build material field
mf = build_material_field(result_fields; use_centroids=false, eltype_out=Float32)

# 6. Run topology optimization
build_KEStore!(dh, mf, nnodes_loc, avg_mp_store)
global u = zeros(ndofs(dh))
X, c = topopt_run(999)  # Use different ID for new run

# 7. Analyze result
frac = sum((0.1 .< X) .& (X .< 0.9)) / length(X)
println("Intermediate density fraction: $(frac*100)%")
```

---

## Troubleshooting

### Problem: "Invalid material field" warnings

**Cause**: Coefficients generating negative material parameters

**Solution**:
1. Reduce σ values: Change `σs` from 0.8 to 0.5 or 0.3
2. Tighten bounds: Change `coeff_bounds=(-2.0, 2.0)` in proxy.jl
3. Check validation: Review diagnostics printed with warning

### Problem: Optimization very slow

**Cause**: Too many modes or TopOpt taking too long

**Solution**:
1. Reduce modes: Set `N_MODES_PER_PROP = 10` or `8`
2. Reduce iterations: Set `max_iterations = 30`
3. Reduce mesh: Edit `nelx`, `nely` in `input/params_geom.jl`

### Problem: Badness not improving

**Cause**: Weights, initialization, or optimizer settings

**Solution**:
1. Adjust weights: Try `w_frac=0.6, w_severity=0.3, w_gray=0.1`
2. More exploration: Increase `initial_sigma=1.0`
3. Check baseline: Run deterministic case first (verify TopOpt works)

### Problem: Different eigenvalues each run

**Cause**: This should NOT happen! Eigenmodes are deterministic.

**Solution**:
- If this occurs, file a bug report
- Check that mesh and parameters are truly identical
- Verify `make_sparse` setting is consistent

---

## Interpreting Results

### Good Optimization Run

```
Iteration  1: Badness = 0.123
Iteration 10: Badness = 0.234
Iteration 20: Badness = 0.345
Iteration 30: Badness = 0.421
Iteration 40: Badness = 0.456
Iteration 50: Badness = 0.467  ← Converging
```

**Characteristics**:
- Badness increases over iterations
- Convergence after 30-40 iterations
- Final intermediate fraction > 30%

### Poor Optimization Run

```
Iteration  1: Badness = 0.087
Iteration 10: Badness = 0.091
Iteration 20: Badness = 0.089
Iteration 30: Badness = 0.092  ← Stagnant
```

**Characteristics**:
- Badness stays low (~0.1 or less)
- No clear trend
- Final intermediate fraction < 15%

**Fixes**: Increase `initial_sigma`, try different weights, check TopOpt baseline

---

## Advanced Usage

### Multi-Start for Global Optimization

Run multiple times with different seeds:

```julia
# In Julia REPL
include("test/proxy.jl")

best_overall = -Inf
best_coeffs_overall = nothing

for i in 1:5
    Random.seed!(42 + i)
    coeffs, badness, result = run_adversarial_optimization(
        max_iterations=30, population_size=0
    )
    
    if badness > best_overall
        best_overall = badness
        best_coeffs_overall = coeffs
    end
end
```

### Analyzing Which Modes Matter Most

After optimization:

```julia
# Load results
data = load("output/adversarial_YYYYMMDD_HHMMSS/optimization_history.jld2")
best_coeffs = data["best_coeffs"]
properties = data["properties"]

# Analyze coefficient magnitudes
for (j, prop) in enumerate(properties)
    coeffs = best_coeffs[:, j]
    active = coeffs[coeffs .!= 0]
    sorted_idx = sortperm(abs.(active), rev=true)
    
    println("\n$prop - Top 5 most important modes:")
    for i in 1:min(5, length(sorted_idx))
        @printf("  Mode %2d: coeff = %+.4f\n", 
                sorted_idx[i], active[sorted_idx[i]])
    end
end
```

---

## Workflow Summary

```
┌─────────────────────────────────────────────────────────────┐
│                    ADVERSARIAL OPTIMIZATION                  │
└─────────────────────────────────────────────────────────────┘

1. PRE-COMPUTATION (once, deterministic)
   ├─ Compute KL eigenmodes for each property
   ├─ Store in kl_modes_dict (FIXED during optimization)
   └─ Total time: ~1-2 minutes

2. OPTIMIZATION LOOP (50 iterations, variable)
   ├─ FOR each iteration:
   │  ├─ Generate material fields from coefficients
   │  ├─ Validate material field
   │  ├─ Run SIMP topology optimization
   │  ├─ Compute badness metrics
   │  ├─ Update coefficients (optimizer)
   │  └─ Log results
   └─ Total time: ~30-60 minutes

3. POST-PROCESSING (automatic)
   ├─ Save best coefficients (TXT)
   ├─ Save optimization history (JLD2)
   ├─ Generate convergence plots (PNG)
   └─ Export checkpoints

4. RECONSTRUCTION (user, as needed)
   ├─ Load best coefficients
   ├─ Re-compute eigenmodes (deterministic)
   ├─ Generate material fields
   ├─ Run TopOpt
   └─ Analyze intermediate densities
```

---

## Quick Reference

| Task | Command | Time |
|------|---------|------|
| Test setup | `julia test/test_adversarial_setup.jl` | 2-5 min |
| Run optimization | `julia test/proxy.jl` | 30-60 min |
| Load results | `load("output/adversarial_*/optimization_history.jld2")` | instant |
| Reconstruct field | See "Reconstructing Best Material Field" above | 2-3 min |

| Parameter | Location | Default | Recommendation |
|-----------|----------|---------|----------------|
| N_modes | proxy.jl:60 | 15 | 10-20 |
| max_iterations | proxy.jl:336 | 50 | 30-100 |
| σ values | proxy.jl:52 | 0.8 | 0.5-1.0 |
| Weights | proxy.jl:243 | 0.4/0.4/0.2 | Tune to taste |

---

## Getting Help

1. **Setup issues**: Run `test/test_adversarial_setup.jl` for diagnostics
2. **Algorithm questions**: Read `docs/ADVERSARIAL_OPTIMIZATION_GUIDE.md`
3. **Implementation details**: See `docs/IMPLEMENTATION_SUMMARY_ADVERSARIAL.md`
4. **KL expansion**: Refer to `docs/KL_seed_usage.md`

---

**Ready to run!** Start with the verification script, then run the full optimization.
