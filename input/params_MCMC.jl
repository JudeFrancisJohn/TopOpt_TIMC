# --- MCMC Configuration ---
const N_CHAIN = 20         # Number of MCMC iterations
const BURN_IN = Int(floor(0.3 * N_CHAIN))  # Burn-in iterations discarded from analysis
const N_MODES = 80          # Number of KL modes (must match what's used in run_single_design/KL_realization)
const PROPOSAL_SIGMA = 0.5 # Step size for random walk proposal
const BETA = 50.0           # Inverse temperature. Higher = stronger preference for "bad" designs.
const DENSITY_WEIGHT = 5.0
const COMPLIANCE_WEIGHT = 1.0e-3