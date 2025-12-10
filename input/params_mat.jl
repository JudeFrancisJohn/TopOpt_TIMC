# Material parameters (transversely isotropic)
λ     = 1.0
μ_l   = 1.0
μ_t   = 1.0
alpha = 1.0
beta  = 1.0
angle = 0.0  # Default fiber angle (can be overridden by multi_angle_topopt.jl)

# Select which material properties are treated as spatial random fields via KL expansion.
# Choose from symbols: :μ_l, :μ_t, :α, :β, :λ, :angle
# First try: vary alpha, beta, and angle
VARIABLE_PROPERTIES = (:α, :β, :angle)
