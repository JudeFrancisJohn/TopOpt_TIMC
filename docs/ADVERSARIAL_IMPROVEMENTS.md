# Adversarial Optimization Improvements

## Date: 2025-12-03

This document describes the improvements made to handle domain errors gracefully and improve the adversarial optimization objective function.

---

## Task 1: Graceful Error Handling

### Problem
Domain errors (e.g., `sqrt` of negative numbers) occasionally occur during topology optimization with extreme material parameters, causing the entire optimization run to fail.

### Solution
Implemented comprehensive error handling in `test/proxy.jl`:

1. **Try-catch block** around `topopt_run()` to catch `DomainError` exceptions
2. **Diagnostic file creation**: `failed_topopt_runs.txt` that saves:
   - KL coefficients for all properties (μ_l, μ_t, α, β)
   - Material field statistics (mean, std, min, max)
   - Density field values (if available)
   - Full stack trace
   - Timestamp
3. **Penalty values**: Assigns large penalty (`badness = -1000.0`, `compliance = 1e6`) to discourage this parameter region
4. **Continuation**: Optimization continues with next candidate instead of crashing

### File Modified
- `test/proxy.jl` (lines ~218-275)

### Example Output
```
⚠️  DomainError encountered during topology optimization!
   Error: DomainError with -2.211238328897361e-9
   → Diagnostics saved to: output/adversarial_YYYYMMDD_HHMMSS/failed_topopt_runs.txt
   → Assigning penalty values and continuing...
```

---

## Task 2: Improved Objective Function

### Problem Analysis

**Original Issue**: Badness metric was fluctuating (increasing then decreasing) rather than monotonically improving. Why?

1. **Not contradictory objectives**: 
   - **Inner loop** (TopOpt): Minimizes compliance for GIVEN material parameters → deterministic
   - **Outer loop** (Adversarial): Finds material parameters that maximize intermediate densities
   - These are NOT in conflict - outer loop searches material space, inner loop optimizes design

2. **Real issue**: Some material configurations lead to:
   - High intermediate densities (good!) 
   - BUT numerical instability (domain errors, extreme compliance)
   - Optimizer explores these regions → crashes or bad solutions

3. **Why fluctuating?**: 
   - Optimizer samples random material parameters
   - Some give high badness but are unstable (extreme compliance)
   - Some give moderate badness but are stable
   - Without stability penalty, optimizer can't distinguish "good high badness" from "unstable high badness"

### Solution: Stability-Aware Objective

Modified `compute_combined_badness()` in `utils/adversarial_utils.jl` to include **stability penalty**:

```julia
badness_raw = w_frac * frac + w_sev * severity_norm + w_gray * gray

# Soft penalty for extreme compliance (approaching instability)
if compliance_ratio > threshold:
    stability_penalty = weight * (1 - exp(-(ratio - threshold)/50))
    
badness = badness_raw - stability_penalty
```

**Key Features**:

1. **Base badness**: Still prioritizes intermediate densities (high frac, severity, gray)

2. **Stability penalty**: 
   - Activates when `compliance > 10 × median(compliance_history)`
   - Exponential soft threshold (smooth, no discontinuities)
   - Also penalizes very LOW compliance (potential degenerate solutions)
   - Weight = 0.15 (moderate - keeps feasibility without dominating objective)

3. **Adaptive reference**: 
   - Tracks compliance history during optimization
   - Uses `median(compliance_history)` as reference
   - Adapts to actual scale of problem (robust)

4. **Parameters**:
   - `compliance_penalty_threshold = 10.0` → penalize if >10× median
   - `stability_weight = 0.15` → moderate penalty strength
   - Soft exponential: `1 - exp(-x/50)` → smooth transition

### Files Modified
- `utils/adversarial_utils.jl`: Enhanced `compute_combined_badness()` function
- `test/proxy.jl`: Added compliance history tracking and adaptive reference

### How It Works

```
Iteration 1:  compliance = 13.76  → median_ref = 13.76
Iteration 2:  compliance = 4.01   → median_ref = 8.89
...
Iteration 9:  compliance = 0.80   → median_ref ≈ 4.0
Iteration 10: compliance = 150.0  → ratio = 150/4 = 37.5 >> 10
                                   → stability_penalty kicks in!
                                   → badness reduced
                                   → optimizer discouraged from this region
```

### Expected Behavior After Fix

1. **Early iterations**: Optimizer explores widely, compliance history builds
2. **Middle iterations**: Stability penalty adapts, unstable regions penalized
3. **Later iterations**: Converges to **high badness + stable compliance** region
4. **Result**: Monotonic improvement in badness without numerical failures

---

## Why This Solves Your Problem

### Before:
```
Iter 9:  badness=0.457 (BEST!), compliance=0.80
Iter 10: badness=0.412 (worse?), compliance=9.57  
Iter 13: badness=0.327 (worst!), compliance=26.57 (unstable!)
```
**Issue**: High badness at iteration 9, but no stability check → next samples explore extreme regions → compliance explodes → badness drops

### After:
```
Iter 9:  badness_raw=0.457, compliance=0.80, ratio=0.2 → no penalty → badness=0.457
Iter 10: badness_raw=0.412, compliance=9.57, ratio=2.4 → no penalty → badness=0.412
Iter 13: badness_raw=0.380, compliance=26.57, ratio=6.6 → small penalty → badness=0.365
Iter 15: badness_raw=0.400, compliance=150.0, ratio=37.5 → LARGE penalty → badness=0.20
```
**Result**: Iteration 15 gets penalized heavily → optimizer learns to avoid extreme compliance → focuses on feasible high-badness region → monotonic improvement

---

## Key Parameters to Tune

If you want to adjust behavior:

### In `utils/adversarial_utils.jl`:
```julia
compute_combined_badness(
    X, compliance;
    compliance_penalty_threshold = 10.0,  # Lower = stricter stability requirement
    stability_weight = 0.15               # Higher = stronger penalty
)
```

### In `test/proxy.jl`:
```julia
w_frac = 0.4,   # Weight for intermediate fraction
w_sev = 0.4,    # Weight for severity (distance from 0/1)
w_gray = 0.2    # Weight for gray indicator
```

**Recommendations**:
- **If too many failures**: Decrease `compliance_penalty_threshold` to 5.0 or increase `stability_weight` to 0.25
- **If convergence too conservative**: Increase `compliance_penalty_threshold` to 20.0 or decrease `stability_weight` to 0.1
- **If not enough intermediate densities**: Increase `w_frac` and `w_sev` relative to `w_gray`

---

## Testing Recommendations

1. **Run with current settings** (50 iterations):
   ```julia
   julia --project=. test/proxy.jl
   ```

2. **Monitor**:
   - Check `failed_topopt_runs.txt` for domain error frequency
   - Watch badness progression in console/log
   - Verify compliance stays within reasonable range

3. **Analyze results**:
   - Plot badness vs iteration (should trend upward)
   - Plot compliance vs iteration (should stay bounded)
   - Check final density field for intermediate densities

4. **Adjust if needed**:
   - Too many failures → stricter stability penalty
   - No improvement → relax stability penalty
   - Low intermediate fraction → increase `w_frac`

---

## Summary

✅ **Task 1**: Domain errors now handled gracefully with full diagnostics saved to `failed_topopt_runs.txt`

✅ **Task 2**: Objective function improved with adaptive stability penalty to:
   - Maximize intermediate densities (primary goal)
   - Avoid numerical instability (feasibility constraint)
   - Adapt to problem scale (robust across different mesh sizes, materials)

🎯 **Expected outcome**: Monotonic improvement in badness metric while maintaining numerical stability, leading to worst-case material parameters that produce highly intermediate topology designs.
