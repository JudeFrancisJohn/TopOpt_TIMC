    # ========================================================================
    # SA PARAMETERS
    # ========================================================================
    const T_INITIAL = 10.0      # Initial temperature (high = more exploration)
    const T_FINAL = 0.01        # Final temperature (low = greedy)
    const COOLING_RATE = (T_FINAL / T_INITIAL)^(1.0 / max_iterations)  # Geometric cooling
    const STEP_SIZE_INITIAL = 1.0  # Initial proposal step size
    const STEP_SIZE_MIN = 0.1      # Minimum step size
    const STEP_SIZE_MAX = 2.0      # Maximum step size
    const ADAPTIVE_WINDOW = 20     # Adjust step size every N iterations
    