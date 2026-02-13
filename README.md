# Penalty-based Topology Optimization for Transversely Isotropic Materials with Stochastic Material parameters

A Finite element codebase to investigate optimization failures in  penalty-based Topology Optimization for transversely isotropic materials.

![Topology Optimization Overview](utils/assets/Topology%20Optimization_presentation2.png)
![Topology Optimization Overview](utils/assets/Topology%20Optimization_presentation2_st.png)
![Topology Optimization Overview](utils/assets/Topology%20Optimization_presentation2_st2.png)
## Overview


Features:
- Ferrite.jl-based finite element solver with nonlinear capabilities
- Transversely isotropic material model with spatially varying parameters
- KL expansion with multiple kernel options (exponential, Gaussian, Matérn)
- SIMP-based topology optimization with OC (Optimality Criteria) updates
- Metropolis-Hastings MCMC for exploring material parameter space
- CMA-ES-based adversarial optimization for worst-case material fields
- VTK export for ParaView visualization
- Modular architecture with pluggable optimization strategies

---

## Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/JudeFrancisJohn/TopOpt_TIMC.git
cd TopOpt_TIMC

# Activate Julia environment
julia --project=.

# Install dependencies
using Pkg
Pkg.instantiate()
```

### Basic Usage

#### 1. Run Stochastic Topology Optimization
```bash
julia --project=. src/COPY_stochastic_modified_v2\ copy\ 2_proxy.jl
```
This runs a single topology optimization with stochastic material parameters.

#### 2. Run Adversarial Optimization (Find Worst-Case Materials)
```bash
julia --project=. test/proxy_new.jl
```
Uses CMA-ES to find KL coefficients that maximize intermediate density regions.

#### 3. Run MCMC Exploration
```bash
julia --project=. test/MC_run_v2.jl
```
Explores material parameter space using Metropolis-Hastings MCMC.

#### 4. Run Multiple Monte Carlo Samples
```julia
include("src/COPY_stochastic_modified_v2_MC.jl")
run_logs = multiple_runs(100)  # 100 independent samples with different seeds
```

---

## Project Structure

```
TopOpt/
├── src/                                      # Main drivers
│   ├── COPY_stochastic_modified_v2 copy 2_proxy.jl   # Main stochastic driver (CURRENT) 
│   ├── COPY_stochastic_modified_v2_MC.jl            # Base driver for Monte Carlo runs
│   ├── geom_BC.jl                                    # Geometry and boundary conditions
│   └── ...                                           # Legacy versions
│
├── test/                                     # Test scripts and experiments
│   ├── proxy_new.jl                         # Adversarial optimization driver 
│   ├── mhmc_run.jl                          # MCMC example (educational)
│   ├── test_adversarial_setup.jl           # Test adversarial components
│   └── unit/                                # Unit tests
│
├── utils/                                    # Core utilities
│   ├── stochastic_utils.jl                  # KL expansion (3-function architecture) 
│   ├── FE_updated_stoch.jl                  # FE solver + material handling 
│   ├── opt.jl                               # OC update + sensitivity filter
│   ├── mcmc_utils.jl                        # MCMC helper functions
│   ├── adversarial_optimizer.jl             # Adversarial optimization wrapper
│   ├── adversarial_objective.jl             # Objective function for adversarial opt
│   ├── adversarial_utils.jl                 # Adversarial utilities
│   ├── adversarial_logging.jl               # Logging for adversarial runs
│   ├── cmaes_optimizer.jl                   # CMA-ES implementation
│   ├── abstract_optimizer.jl                # Optimizer interface
│   ├── io_manager.jl                        # I/O utilities
│   ├── metrics.jl                           # Design quality metrics
│   ├── validators.jl                        # Material field validation
│   └── diagnostic_utils.jl                  # Debugging utilities
│
├── input/                                    # Configuration parameters
│   ├── params_mat.jl                        # Material parameters 
│   ├── params_geom.jl                       # Geometry (mesh, dimensions)
│   ├── params_topopt.jl                     # Topology optimization settings
│   ├── params_adversarial.jl                # Adversarial optimization config
│   ├── params_MCMC.jl                       # MCMC configuration
│   ├── params_LOGS.jl                       # Logging settings
│   └── params_SimulatedAnnealing.jl         # Simulated annealing config
│
├── output/                                   # Results (auto-generated)
│   ├── stochastic_YYYYMMDD_HH/              # Stochastic run results
│   ├── adversarial_YYYYMMDD_HHMM/           # Adversarial optimization results
│   ├── mcmc_chain_YYYYMMDD_HHMMSS.jld2      # MCMC chain data
│   └── eigenmode_cache/                     # Cached KL eigenmodes
│
├── tools/                                    # Helper scripts
│   └── quick_vtk_test.jl                    # VTK export testing
│
├── Project.toml                              # Julia package dependencies
├── Manifest.toml                             # Locked dependency versions
└── README.md                                 # This file
```

---

## Mathematical Background

### KL Expansion
Material fields are represented using Karhunen-Loève expansion:

$$\theta(x) = \mu + \sum_{i=1}^{N} \sqrt{\lambda_i} \xi_i \phi_i(x)$$

- **μ**: Mean value (from `MaterialParams`)
- **λᵢ, φᵢ**: Eigenmodes (deterministic, computed once from covariance structure)
- **ξᵢ**: Coefficients (stochastic, sampled from N(0,1) or explored via MCMC/adversarial opt)

**Modes:**
- `:lognormal`: Ensures positivity, `field = μ × exp(gaussian)`
- `:additive`: Allows negative values, `field = μ + gaussian`

### Transversely Isotropic Material
Constitutive model with 6 independent parameters:
- **λ**: Lamé parameter
- **μ_l**: Longitudinal shear modulus
- **μ_t**: Transverse shear modulus
- **α, β**: Anisotropy parameters
- **angle**: Fiber orientation angle

### SIMP Topology Optimization
- **Penalty**: Material stiffness scaled by `x^p` where `x` is element density
- **OC Update**: Optimality Criteria method with move limits
- **Sensitivity Filter**: Density filter to avoid checkerboard patterns

---

## Key Components

### 1. KL Expansion (utils/stochastic_utils.jl)

**Three-Function Architecture:**

```julia
# 1. Pre-compute eigenmodes (ONCE per property)
kl_modes = compute_KL_eigenmodes(mp, coords_elem, :μ_l, sigma;
    Lc=0.01, N_modes=80, kernel=:exponential, mode=:lognormal)

# 2. Sample field from coefficients (FAST, many times)
coeffs = randn(80)  # or from MCMC/adversarial optimizer
field = sample_KL_field(kl_modes, coeffs)

# 3. Legacy wrapper (one-shot, less efficient)
fields = KL_realization(mp, coords_elem; 
    σs=Dict(:μ_l => 0.1), Lc=0.01, N_modes=80, seed=42)
```

**Available Kernels:**
- `:exponential`: C(r) = σ² exp(-r/Lc)
- `:gaussian`: C(r) = σ² exp(-r²/(2Lc²))
- `:matern`: Matérn covariance with ν = 1.5 or 2.5

### 2. Material Field Types (utils/FE_updated_stoch.jl)

```julia
# Scalar mean parameters
MaterialParams(λ, μ_l, μ_t, alpha, beta, angle)

# Spatially varying field (per-element or per-node)
MaterialField{T}(μ_l, μ_t, α, β, λ, angle; use_centroids::Bool)
```

### 3. Topology Optimization Loop

```julia
while change > 0.01 && loop < maxloop
    # 1. FE solve with current densities
    u, λf = FE_Run!(ℂ, x_fe)
    
    # 2. Compute compliance and sensitivities
    c, dc = compute_compliance_and_sensitivities(...)
    
    # 3. Apply sensitivity filter
    dc = check(nelx, nely, rmin, x, dc)
    
    # 4. Update densities using OC
    xnew = OC(x, volfrac, dc)
    
    # 5. Check convergence
    change = maximum(abs.(xnew .- x))
    x = xnew
    loop += 1
end
```

### 4. [!Obsolete] MCMC Exploration (test/MC_run_v2.jl) [!Obsolete]

Finds KL coefficients that produce designs with high intermediate densities (considered "bad" or smudged designs).

**Badness Metric:**
```julia
badness = DENSITY_WEIGHT * (proportion of elements with 0.3 < x < 0.7) 
        + COMPLIANCE_WEIGHT * compliance
```

**[!Obsolete] Metropolis-Hastings Algorithm:**
```julia
# Propose new coefficients
proposed_coeffs = current_coeffs + randn(N_modes) * PROPOSAL_SIGMA

# Run TopOpt with proposed material field
log_entry = run_with_kl_coeffs(run_i, proposed_coeffs, kl_modes_dict)

# Accept/reject based on badness
acceptance_prob = min(1.0, exp(BETA * (proposed_badness - current_badness)))
if rand() < acceptance_prob
    current_coeffs = proposed_coeffs  # Accept
end
```

### 5. Adversarial Optimization (test/proxy_new.jl)

Finds worst-case material fields using CMA-ES (gradient-free optimization).

**CMA-ES Configuration:**
```julia
optimizer = CMAESOptimizer(
    n_modes_dict, properties,
    max_iterations=MAX_ITERATIONS,
    population_size=POPULATION_SIZE,
    initial_sigma=INITIAL_SIGMA,
    w_frac=W_INTERMEDIARY_FRAC,
    w_severity=W_INTERMEDIARY_SEVERITY,
    w_gray=W_GRAYNESS,
    save_path=SAVE_PATH
)

# Run optimization
best_coeffs, best_score, history = optimize!(optimizer, objective_fn)
```

---

## Configuration

### Material Parameters (input/params_mat.jl)
```julia
# Mean material properties
λ     = 1.0    # Lamé parameter
μ_l   = 1.0    # Longitudinal shear modulus
μ_t   = 1.0    # Transverse shear modulus
alpha = 1.0    # Anisotropy parameter
beta  = 1.0    # Anisotropy parameter
angle = 0.01   # Fiber angle (radians)

# Properties to treat as random fields
VARIABLE_PROPERTIES = (:α, :β, :angle)
```

### Geometry (input/params_geom.jl)
```julia
lx    = 30     # Domain length
ly    = 8      # Domain height
nelx  = 60     # Elements in x-direction
nely  = 20     # Elements in y-direction
```

### Topology Optimization (input/params_topopt.jl)
```julia
volfrac = 0.5  # Target volume fraction
penal   = 3.0  # SIMP penalty exponent
rmin    = 1.5  # Sensitivity filter radius
nruns   = 2    # Number of runs for multiple_runs()
```

### Adversarial Optimization (input/params_adversarial.jl)
```julia
const ADVERSARIAL_SEED = 42
const N_MODES_PER_PROP = 15  # KL modes per property

# Standard deviations for KL expansion
const σs_ADVERSARIAL = Dict(
    :α   => 1.5,     # ±80% variation
    :β   => 1.5,     # ±80% variation
    :angle => 30.0,  # ±45° range (3σ)
)

# Optimization settings
const MAX_ITERATIONS = 100
const POPULATION_SIZE = 20
const INITIAL_SIGMA = 1.0
```

### [!Obsolete] MCMC (input/params_MCMC.jl)
```julia
const N_CHAIN = 20              # MCMC iterations
const BURN_IN = 6               # Burn-in period
const PROPOSAL_SIGMA = 0.5      # Step size for proposals
const BETA = 50.0               # Inverse temperature
const DENSITY_WEIGHT = 5.0      # Weight for intermediate densities
const COMPLIANCE_WEIGHT = 1.0e-3
```

---

## Output and Results

### Stochastic Runs
Location: `output/stochastic_YYYYMMDD_HH/`

Contains:
- VTK files for each iteration
- Final design density field
- Material field distributions

### Adversarial Optimization
Location: `output/adversarial_YYYYMMDD_HHMM/`

Contains:
- `best_coefficients.txt` - Optimal KL coefficients
- `best_metadata.txt` - Best objective value and iteration
- `convergence.png` - Optimization history plot
- `optimization_log.txt` - Detailed iteration log
- `adversarial_results.jld2` - Full results (coefficients, history, best design)
- VTK files for best design

### [!Obsolete] MCMC Chains
Location: `output/mcmc_chain_YYYYMMDD_HHMMSS.jld2`

Contents:
```julia
data = load("output/mcmc_chain_*.jld2")
chain_records = data["chain_records"]  # Full iteration history
chain_scores = data["chain_scores"]    # Badness scores
coeffs_history = data["coeffs_history"] # Coefficient evolution
```

### Visualization
VTK files can be opened in ParaView to view:
- Density fields
- Displacement fields
- Material parameter distributions
- Stress/strain fields

---

## Common Workflows

### Debug FEA Solver
```julia
include("src/COPY_stochastic_modified_v2 copy 2_proxy.jl")
# Look for "FEA verification" message early in output
```

### Test Adversarial Setup
```julia
julia --project=. test/test_adversarial_setup.jl
# Verifies eigenmode computation, field generation, and objective evaluation
```

### [!Obsolete] Adjust MCMC Acceptance Rate
Target: 20-40% acceptance rate

Too low (<10%):
- Reduce `PROPOSAL_SIGMA` (try 0.1 or 0.05)
- Check if `BETA` is too high

Too high (>90%):
- Increase `PROPOSAL_SIGMA` (try 1.0 or 2.0)
- Check if `BETA` is too low

### Visualize Results in ParaView
```bash
# Open ParaView
paraview output/stochastic_*/final/*.vtu
```

### Change Problem Geometry
Edit `input/params_geom.jl`:
```julia
nelx, nely = 120, 40  # Finer mesh
lx, ly = 60, 16       # Larger domain
```

---

## Testing

Run unit tests:
```bash
julia --project=. test/unit/runtests.jl
```

Run convergence test:
```bash
julia --project=. test/unit/test_convergence_plot.jl
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{topopt_stochastic_2026,
  title = {A study on penalty-based topology Optimization of transversely isotropic materials with stochastic material parameters},
  author = {Jude Francis},
  year = {2026},
  url = {https://github.com/JudeFrancisJohn/TopOpt_TIMC}
}
```

---

## License

This work is licensed under a [Creative Commons Attribution-NonCommercial 4.0 International License](http://creativecommons.org/licenses/by-nc/4.0/).

[![CC BY-NC 4.0](https://licensebuttons.net/l/by-nc/4.0/88x31.png)](http://creativecommons.org/licenses/by-nc/4.0/)

## Contact

 
**GitHub:** [@JudeFrancisJohn](https://github.com/JudeFrancisJohn)

---

## Acknowledgments

This work was completed as part of the Seminar Research Project in MSc Computational Mechanics of Materials and Structures (COMMAS).

- Ferrite.jl for the finite element framework
- Julia community for the ecosystem of scientific computing packages

---

**Version:** 3.0  
**Last Updated:** February 7, 2026  
**Status:** Active Development
