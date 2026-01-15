"""
params_adversarial.jl

Configuration parameters for adversarial topology optimization.
This file contains all constant parameters, random seeds, and optimization settings
for the adversarial search using KL expansion coefficients.
"""

using Random

# ============================================================================
# RANDOM SEED
# ============================================================================
const ADVERSARIAL_SEED = 42

# ============================================================================
# KL EXPANSION PARAMETERS
# ============================================================================

# Number of KL modes per material property
const N_MODES_PER_PROP = 15  # Reduced for computational efficiency

# Standard deviations for each material property in KL expansion
const σs_ADVERSARIAL = Dict(
    :μ_l => 0.5,
    :μ_t => 0.5,
    :α   => 1.5,
    :β   => 1.5,
    :λ   => 0.5,
    :angle => 2.0,
)

# Correlation lengths for KL expansion (spatial scale of variation)
# Different properties can have different spatial correlation structures
# Note: Using a function to avoid constant redefinition issues in REPL
function get_Lc_ADVERSARIAL(prop_sym::Symbol)
    lc_dict = Dict(
        :μ_l => 2.0,    # Longitudinal shear: medium-scale variations
        :μ_t => 1.5,    # Transverse shear: finer-scale variations
        :α   => 3.0,    # Alpha: coarser variations
        :β   => 3.0,    # Beta: coarser variations
        :λ   => 2.5,    # Lambda: medium-coarse variations
        :angle => 4.0,  # Angle: very coarse variations
    )
    return get(lc_dict, prop_sym, 2.0)  # Default to 2.0 if property not found
end

# ============================================================================
# OPTIMIZATION PARAMETERS
# ============================================================================

# Maximum number of optimization iterations
const MAX_ITERATIONS_ADVERSARIAL = 50

# Population size for evolutionary algorithms (0 = auto-determine)
const POPULATION_SIZE_ADVERSARIAL = 0

# Initial step size / sigma for CMA-ES
const INITIAL_SIGMA_ADVERSARIAL = 0.5

# Coefficient bounds for KL expansion coefficients
const COEFF_LOWER_BOUND = -3.0
const COEFF_UPPER_BOUND = 3.0

# ============================================================================
# OBJECTIVE FUNCTION WEIGHTS
# ============================================================================

# Weight for intermediate density fraction metric
const W_FRAC_ADVERSARIAL = 0.4

# Weight for intermediate density severity metric
const W_SEVERITY_ADVERSARIAL = 0.4

# Weight for gray indicator metric
const W_GRAY_ADVERSARIAL = 0.2

# Stability weight for compliance penalty
const W_STABILITY_ADVERSARIAL = 0.15

# Compliance penalty threshold (relative to median)
const COMPLIANCE_PENALTY_THRESHOLD = 10.0

# ============================================================================
# LOGGING AND OUTPUT
# ============================================================================

# Save checkpoint every N iterations
const SAVE_EVERY_ADVERSARIAL = 10

# Material field validation tolerance factor (higher = more permissive)
const VALIDATION_TOLERANCE_FACTOR = 10.0

# ============================================================================
# THRESHOLDS FOR INTERMEDIATE DENSITY DETECTION
# ============================================================================

const THRESHOLD_LOW_ADVERSARIAL = 0.1
const THRESHOLD_HIGH_ADVERSARIAL = 0.9

# ============================================================================
# OPTIMIZER SELECTION
# ============================================================================

# Available optimizer types: :cmaes, :simulated_annealing, :adaptive_de
const OPTIMIZER_TYPE = :cmaes

println("Loaded adversarial optimization parameters:")
println("  Optimizer type: $(OPTIMIZER_TYPE)")
println("  Max iterations: $(MAX_ITERATIONS_ADVERSARIAL)")
println("  KL modes per property: $(N_MODES_PER_PROP)")
println("  Random seed: $(ADVERSARIAL_SEED)")
