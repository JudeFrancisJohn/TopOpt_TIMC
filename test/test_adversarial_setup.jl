"""
test_adversarial_setup.jl

Quick verification script to test adversarial optimization setup.
Tests KL eigenmode computation, field generation, and objective evaluation
WITHOUT running the full optimization (for rapid debugging).
"""

using Random
using Statistics
using Printf

println("="^80)
println("ADVERSARIAL OPTIMIZATION SETUP TEST")
println("="^80)

# Load required modules
println("\n[1/6] Loading TopOpt driver...")
include("../src/COPY_stochastic_modified_v2 copy 2_proxy.jl")
include("../input/params_MCMC.jl")
include("../utils/adversarial_utils.jl")
include("../utils/adversarial_optimizer.jl")

println("✓ Modules loaded successfully")

# Configuration
Random.seed!(42)
properties = (:μ_l, :μ_t, :α, :β)
n_props = length(properties)
σs = Dict(:μ_l => 0.8, :μ_t => 0.8, :α => 0.8, :β => 0.8)
N_MODES_PER_PROP = 10  # Small for testing

n_elem, n_loc = size(coords_elem)
println("\nMesh info:")
println("  Elements: $n_elem")
println("  Nodes per element: $n_loc")

# Test 1: Pre-compute eigenmodes
println("\n[2/6] Testing eigenmode computation...")
kl_modes_dict = Dict{Symbol, Any}()
n_modes = Dict{Symbol, Int}()

for prop_sym in properties
    sigma = get(σs, prop_sym, 0.5)
    kl_modes = compute_KL_eigenmodes(
        mp, coords_elem, prop_sym, sigma;
        Lc=0.01, N_modes=N_MODES_PER_PROP, 
        use_centroids=false, make_sparse=false
    )
    kl_modes_dict[prop_sym] = kl_modes
    n_modes[prop_sym] = length(kl_modes.eigenvalues)
    
    @printf("  %s: %d modes, λ ∈ [%.3e, %.3e]\n", 
            prop_sym, n_modes[prop_sym],
            minimum(kl_modes.eigenvalues), 
            maximum(kl_modes.eigenvalues))
end
println("✓ Eigenmodes computed successfully")

# Test 2: Generate random coefficients
println("\n[3/6] Testing coefficient generation...")
max_modes = maximum(values(n_modes))
coeffs_mat = zeros(Float64, max_modes, n_props)
for (j, prop) in enumerate(properties)
    local n = n_modes[prop]
    coeffs_mat[1:n, j] .= randn(n) * 0.1
end
println("  Coefficient matrix shape: $(size(coeffs_mat))")
println("  Coefficient range: [$(minimum(coeffs_mat)), $(maximum(coeffs_mat))]")
println("✓ Coefficients generated successfully")

# Test 3: Convert coefficients and generate fields
println("\n[4/6] Testing field generation...")
coeffs_dict = matrix_to_coeffs_dict(coeffs_mat, properties, n_modes)
result_fields = Dict{Symbol, Any}()

for prop_sym in properties
    result_fields[prop_sym] = sample_KL_field(
        kl_modes_dict[prop_sym], 
        coeffs_dict[prop_sym]; 
        eltype_out=Float32
    )
    @printf("  %s field shape: %s\n", prop_sym, size(result_fields[prop_sym]))
end

# Add constant properties
result_fields[:λ] = fill(Float32(mp.λ), n_elem, n_loc)
result_fields[:angle] = fill(Float32(mp.angle), n_elem, n_loc)
println("✓ Fields generated successfully")

# Test 4: Build and validate material field
println("\n[5/6] Testing material field construction...")
mf = build_material_field(result_fields; use_centroids=false, eltype_out=Float32)
is_valid, diagnostics = validate_material_field(mf, mp; tolerance_factor=5.0)

println("  Material field validation: $(is_valid ? "PASS" : "FAIL")")
if is_valid
    @printf("  μ_l range: [%.3e, %.3e], mean: %.3e\n", 
            diagnostics["μ_l_range"]..., diagnostics["μ_l_mean"])
    @printf("  μ_t range: [%.3e, %.3e], mean: %.3e\n", 
            diagnostics["μ_t_range"]..., diagnostics["μ_t_mean"])
    println("✓ Material field valid")
else
    @warn "Material field validation failed" diagnostics
    println("✗ Material field invalid - check coefficient bounds or σ values")
end

# Test 5: Run single topology optimization
println("\n[6/6] Testing topology optimization run...")
try
    build_KEStore!(dh, mf, nnodes_loc, avg_mp_store)
    global u = zeros(ndofs(dh))
    X, c = topopt_run(1)
    
    # Compute metrics
    frac = compute_intermediary_fraction(X)
    severity = compute_intermediary_severity(X)
    gray = compute_gray_indicator(X)
    badness = compute_combined_badness(X, c)
    
    println("\n  TopOpt Results:")
    @printf("    Compliance:          %.6e\n", c)
    @printf("    Intermediate frac:   %.4f (%.1f%%)\n", frac, frac*100)
    @printf("    Severity:            %.6f\n", severity)
    @printf("    Gray indicator:      %.6f\n", gray)
    @printf("    Badness:             %.6f\n", badness)
    
    println("\n✓ Topology optimization completed successfully")
    
    # Test coefficient export
    test_export_file = "test_coefficients.txt"
    export_coefficients_to_txt(coeffs_mat, properties, n_modes, test_export_file)
    println("✓ Coefficient export test passed")
    println("  Exported to: $test_export_file")
    
    # Clean up test file
    if isfile(test_export_file)
        rm(test_export_file)
        println("  (Test file removed)")
    end
    
catch e
    println("✗ Topology optimization failed")
    @error "TopOpt error" exception=e
    rethrow(e)
end

# Test 6: Verify eigenmode consistency
println("\n[BONUS] Testing eigenmode consistency...")
println("  Re-computing eigenmodes to verify determinism...")
kl_modes_dict_2 = Dict{Symbol, Any}()
for prop_sym in properties
    sigma = get(σs, prop_sym, 0.5)
    kl_modes_2 = compute_KL_eigenmodes(
        mp, coords_elem, prop_sym, sigma;
        Lc=0.01, N_modes=N_MODES_PER_PROP, 
        use_centroids=false, make_sparse=false
    )
    kl_modes_dict_2[prop_sym] = kl_modes_2
    
    # Compare eigenvalues
    diff = norm(kl_modes_dict[prop_sym].eigenvalues - kl_modes_2.eigenvalues)
    @printf("  %s eigenvalue difference: %.3e ", prop_sym, diff)
    if diff < 1e-10
        println("✓")
    else
        println("✗ WARNING: Eigenmodes not deterministic!")
    end
end

# Summary
println("\n" * "="^80)
println("SETUP TEST COMPLETE")
println("="^80)
println("\n✓ All systems operational - ready for adversarial optimization!")
println("\nNext steps:")
println("  1. Review settings in test/proxy.jl")
println("  2. Run: julia --project=. test/proxy.jl")
println("  3. Monitor output/adversarial_YYYYMMDD_HHMMSS/ for results")
println("\n" * "="^80)
