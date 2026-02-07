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
# For lognormal mode: controls multiplicative variation (CoV ≈ σ for small σ)
# For additive mode (angle): controls absolute variation in degrees
const σs_ADVERSARIAL = Dict(
    :μ_l => 0.5,      # Lognormal: ±50% variation around mean
    :μ_t => 0.5,      # Lognormal: ±50% variation around mean
    :α   => 1.5,      # Lognormal: ±80% variation (reduced from 1.5 for stability)
    :β   => 1.5,      # Lognormal: ±80% variation (reduced from 1.5 for stability)
    :λ   => 0.5,      # Lognormal: ±50% variation around mean
    :angle => 30.0,   # Additive: ±45° range (3σ) - reduced from 30° for stability
)

# KL expansion mode for each property
# :lognormal ensures positivity (field = mean × exp(gaussian))
# :additive allows negative values (field = mean + gaussian)
const KL_MODES_ADVERSARIAL = Dict(
    :μ_l => :lognormal,
    :μ_t => :lognormal,
    :α   => :lognormal,
    :β   => :lognormal,
    :λ   => :lognormal,
    :angle => :additive,
)

# Correlation lengths for KL expansion (spatial scale of variation)
# Different properties can have different spatial correlation structures
# Note: Using a function to avoid constant redefinition issues in REPL
function get_Lc_ADVERSARIAL(prop_sym::Symbol)
"""    lc_dict = Dict(
        :μ_l => 2.0,    # Longitudinal shear: medium-scale variations
        :μ_t => 1.5,    # Transverse shear: finer-scale variations
        :α   => 3.0,    # Alpha: coarser variations
        :β   => 3.0,    # Beta: coarser variations
        :λ   => 2.5,    # Lambda: medium-coarse variations
        :angle => 5.0,  # Angle: very coarse variations
    )"""
    lc_dict = Dict(
        :μ_l => 5.0,    # Longitudinal shear: medium-scale variations
        :μ_t => 5.5,    # Transverse shear: finer-scale variations
        :α   => 5.0,    # Alpha: coarser variations
        :β   => 5.0,    # Beta: coarser variations
        :λ   => 5.0,    # Lambda: medium-coarse variations
        :angle => 5.0,  # Angle: very coarse variations
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

# Available optimizer types: :blackboxoptim, :simulated_annealing
const OPTIMIZER_TYPE = :blackboxoptim

# BlackBoxOptim method selection
# Available methods: :adaptive_de_rand_1_bin_radiuslimited (default),
#   :adaptive_de_rand_1_bin, :xnes, :dxnes, :separable_nes,
#   :de_rand_1_bin, :de_rand_2_bin, :generating_set_search,
#   :probabilistic_descent, :resampling_memetic_search, etc.
# See BlackBoxOptim.jl docs for full list
const BBOPTIM_METHOD = :adaptive_de_rand_1_bin_radiuslimited #xnes

println("Loaded adversarial optimization parameters:")
println("  Optimizer type: $(OPTIMIZER_TYPE)")
if OPTIMIZER_TYPE == :blackboxoptim
    println("  BlackBoxOptim method: $(BBOPTIM_METHOD)")
end
println("  Max iterations: $(MAX_ITERATIONS_ADVERSARIAL)")
println("  KL modes per property: $(N_MODES_PER_PROP)")
println("  Random seed: $(ADVERSARIAL_SEED)")
