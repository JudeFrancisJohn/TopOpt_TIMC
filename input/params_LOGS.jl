# Logging and evaluation parameters for topology optimization runs
# Histogram bins for density field evaluation
const LOG_BIN_LOW = 0.15
const LOG_BIN_MID = 0.75
const LOG_BIN_HIGH = 1.0

# Threshold for flagging "bad" designs based on the middle bin proportion
const LOG_BAD_BIN_THRESHOLD = 0.5

# Controls for enforcing shear modulus variability across stochastic samples
const LOG_ENFORCE_SHEAR_VARIATION = true
const LOG_SHEAR_COV_THRESHOLD = 35.0         # max allowable coefficient of variation for μ_l or μ_t
const LOG_SHEAR_MAX_RESAMPLES = 20             # attempts to resample before accepting the best available field
const LOG_ABORT_ON_SHEAR_FAILURE = true        # when true, skip runs that never satisfy the COV bound
