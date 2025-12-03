# TopOpt - Stochastic Material Parameter Topology Optimization

MCMC-based exploration of material parameter fields that lead to poor topology optimization designs.

## Quick Start

### Run MCMC to Find Bad Designs
```bash
julia --project=. test/MC_run_v2.jl
```

This will:
1. Pre-compute KL expansion eigenmodes (deterministic, ~50s)
2. Run Metropolis-Hastings MCMC to explore coefficient space
3. Save results to `output/mcmc_chain_YYYYMMDD_HHMMSS.jld2`

### Run Independent Monte Carlo Samples
```julia
include("src/COPY_stochastic_modified_v2_MC.jl")
run_logs = multiple_runs(100)  # 100 independent samples
```

## Key Concepts

### KL Expansion
Material fields are represented as:

$$\theta(x) = \mu + \sum_{i=1}^{N} \sqrt{\lambda_i} \xi_i \phi_i(x)$$

- **μ**: Mean value (from `MaterialParams`)
- **λᵢ, φᵢ**: Eigenmodes (deterministic - computed once)
- **ξᵢ**: Coefficients (stochastic - what MCMC explores)

### MCMC Goal
Find coefficient vectors {ξᵢ} that produce material parameter fields leading to topology-optimized designs with high intermediary densities (0.3-0.7), which are considered "bad" or smudged designs.

## Project Structure

```
src/
  ├─ COPY_stochastic_modified_v2_MC.jl  # Base driver (TopOpt + FEA)
  └─ geom_BC.jl                         # Geometry and boundary conditions

test/
  ├─ MC_run_v2.jl                       # Main MCMC driver 
  └─ Simple MC_run.jl                   # Legacy (deprecated)

utils/
  ├─ stochastic_utils.jl                # KL expansion (3-function architecture)
  ├─ FE_updated_stoch.jl                # FE routines + material handling
  ├─ opt.jl                             # OC update + sensitivity filter
  └─ mcmc_utils.jl                      # MCMC helper functions

input/
  ├─ params_mat.jl                      # Material parameters
  ├─ params_geom.jl                     # Geometry parameters
  ├─ params_topopt.jl                   # Topology optimization parameters
  └─ params_LOGS.jl                     # Logging parameters

docs/
  ├─ MCMC_and_KL_Guide.md              # Comprehensive usage guide 
  ├─ IMPLEMENTATION_UPDATE.md          # Recent changes summary
  └─ KL_seed_usage.md                  # Seed/reproducibility guide
```

## Main Functions

### KL Expansion (utils/stochastic_utils.jl)

**Efficient (for MCMC):**
```julia
# 1. Pre-compute eigenmodes (once)
kl_modes = compute_KL_eigenmodes(mp, coords_elem, :μ_l, sigma;
    Lc=0.01, N_modes=80, kernel=:exponential, mode=:lognormal)

# 2. Sample many times
for i in 1:N_samples
    coeffs = randn(80)  # or from MCMC proposal
    field = sample_KL_field(kl_modes, coeffs)
end
```

**Simple (for one-off samples):**
```julia
fields = KL_realization(mp, coords_elem;
    σs=Dict(:μ_l => 0.1*mp.μ_l, :μ_t => 0.1*mp.μ_t),
    Lc=0.01, N_modes=80, seed=42)
```

### MCMC (test/MC_run_v2.jl)

```julia
# Configuration
const N_CHAIN = 20              # MCMC iterations
const PROPOSAL_SIGMA = 0.5      # Step size
const BETA = 50.0               # Inverse temperature
const DENSITY_WEIGHT = 5.0      # Weight for intermediary densities

# Badness metric (higher = worse design)
badness = DENSITY_WEIGHT * (intermediary_density_proportion) 
        + COMPLIANCE_WEIGHT * compliance
```

## Configuration

### Material Parameters (input/params_mat.jl)
```julia
λ = 1.0        # Lamé parameter
μ_l = 1.0      # Longitudinal shear modulus
μ_t = 0.5      # Transverse shear modulus
alpha = 0.3    # Anisotropy parameter
beta = 0.2     # Anisotropy parameter
angle = 0.0    # Fiber angle
```

### KL Expansion (test/MC_run_v2.jl)
```julia
const KL_SIGMA_DICT = Dict(
    :μ_l => 0.1 * mp.μ_l,      # 10% COV
    :μ_t => 0.1 * mp.μ_t,      # 10% COV
    :α   => 0.1 * mp.alpha,    # 10% COV
    :β   => 0.1 * mp.beta      # 10% COV
)
const KL_LC = 0.01              # Correlation length
const KL_KERNEL = :exponential  # Covariance kernel
const N_MODES = 80              # Number of modes
```

### Topology Optimization (input/params_topopt.jl)
```julia
nelx, nely = 60, 20      # Mesh size
volfrac = 0.5            # Volume fraction
penal = 3.0              # SIMP penalty
rmin = 1.5               # Filter radius
```

## Output

### MCMC Results
- Location: `output/mcmc_chain_YYYYMMDD_HHMMSS.jld2`
- Contains:
  - `chain_records`: Full iteration history
  - `chain_scores`: Badness scores
  - `post_burnin_scores`: Post burn-in scores
  - Configuration parameters

### VTK Files (if enabled)
- Location: `output/mcmc_chain_*/run_*/final/*.vtu`
- Can be opened in ParaView for visualization

## Performance

### MCMC Efficiency
- **Pre-computation**: ~50s (once per chain)
- **Per iteration**: ~10s (using pre-computed eigenmodes)
- **Speedup vs old approach**: 4-6× faster

### Typical Runtimes
- 20 iterations: ~4 minutes
- 100 iterations: ~18 minutes
- 1000 iterations: ~3 hours

## Documentation

📖 **[MCMC and KL Guide](docs/MCMC_and_KL_Guide.md)** - Complete usage guide  
📄 **[Implementation Update](docs/IMPLEMENTATION_UPDATE.md)** - Recent changes  


## Dependencies

### Core
- Julia 1.11.5
- Ferrite.jl (FEM)
- LinearAlgebra, SparseArrays
- Arpack (sparse eigenvalue problems)

### I/O and Utilities
- WriteVTK (visualization)
- JLD2 (data storage)
- Plots, Statistics, Dates

## Common Workflows

### Debug FEA Only
```julia
include("src/COPY_stochastic_modified_v2 copy 2.jl")
# Look for "FEA verification" message
```

### Adjust MCMC Parameters
Edit `test/MC_run_v2.jl`:
- `PROPOSAL_SIGMA`: Tune acceptance rate (target 20-40%)
- `BETA`: Higher values prefer worse designs
- `N_CHAIN`: More iterations = better exploration

### Change Problem Setup
- Geometry: Edit `input/params_geom.jl`
- Material: Edit `input/params_mat.jl`
- Optimization: Edit `input/params_topopt.jl`

## Troubleshooting

### Low MCMC Acceptance Rate (<10%)
- Reduce `PROPOSAL_SIGMA` (try 0.1 or 0.05)
- Check if BETA is too high

### High MCMC Acceptance Rate (>90%)
- Increase `PROPOSAL_SIGMA` (try 1.0 or 2.0)
- Check if BETA is too low

### Arpack Errors
- Set `make_sparse=false` in KL configuration
- Reduce `N_MODES`
- Check that Arpack.jl is installed

### Memory Issues
- Reduce mesh size (`nelx`, `nely`)
- Use `use_centroids=true` for smaller fields
- Set `make_sparse=true`



**Version**: 2.0  
**Last Updated**: November 22, 2025  

