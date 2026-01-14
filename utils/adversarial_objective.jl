"""
adversarial_objective.jl

Objective function evaluation for adversarial optimization.
Generates material fields from KL coefficients and runs topology optimization.
"""

using Statistics

include("adversarial_utils.jl")
include("adversarial_logging.jl")

# ============================================================================
# OBJECTIVE FUNCTION
# ============================================================================

"""
    evaluate_objective(coeffs_mat::Matrix{Float64}, kl_modes_dict::Dict, properties::Tuple,
                      n_modes::Dict, mp, coords_elem, dh, mf_global, avg_mp_store, 
                      topopt_run_fn::Function, validation_tolerance::Float64)

Evaluate adversarial objective for given coefficient matrix.

# Arguments
- `coeffs_mat::Matrix{Float64}`: Coefficient matrix (max_modes × n_props)
- `kl_modes_dict::Dict`: Pre-computed KL eigenmodes for each property
- `properties::Tuple`: Property symbols being optimized
- `n_modes::Dict`: Number of modes per property
- `mp`: Mean material parameters
- `coords_elem`: Element coordinates for KL field generation
- `dh`: DofHandler for FE assembly
- `mf_global`: MaterialField to update (mutable)
- `avg_mp_store`: Storage for averaged material parameters
- `topopt_run_fn::Function`: TopOpt driver function with signature `f(run_id) -> (X, compliance)`
- `validation_tolerance::Float64`: Tolerance factor for material field validation

# Returns
Tuple of (badness, compliance, X, diagnostics)
"""
function evaluate_objective(coeffs_mat::Matrix{Float64}, 
                           kl_modes_dict::Dict, 
                           properties::Tuple,
                           n_modes::Dict,
                           mp,
                           coords_elem,
                           dh,
                           mf_global,
                           avg_mp_store,
                           topopt_run_fn::Function,
                           build_KE_fn::Function;
                           validation_tolerance::Float64=10.0,
                           output_root::String="output")
    
    N_ELEM, N_LOC = size(coords_elem)
    
    # Convert matrix to Dict format
    coeffs_dict = matrix_to_coeffs_dict(coeffs_mat, properties, n_modes)
    
    # Generate KL fields
    result_fields = Dict{Symbol, Any}()
    for prop_sym in properties
        coeffs_for_prop = coeffs_dict[prop_sym]
        field = sample_KL_field(
            kl_modes_dict[prop_sym], 
            coeffs_for_prop; 
            eltype_out=Float32
        )
        result_fields[prop_sym] = field
    end
    
    # Add constant properties not varied by KL
    for prop_sym in (:μ_l, :μ_t, :α, :β, :λ, :angle)
        if !in(prop_sym, properties)
            val = getproperty(mp, prop_sym)
            result_fields[prop_sym] = fill(Float32(val), N_ELEM, N_LOC)
        end
    end
    
    # Build material field
    mf = build_material_field(result_fields; use_centroids=false, eltype_out=Float32)
    
    # Validate material field (relaxed for adversarial search)
    is_valid, diagnostics = validate_material_field(mf, mp; tolerance_factor=validation_tolerance)
    
    if !is_valid
        has_negative_mu = get(diagnostics, "μ_l_negative", false) || 
                          get(diagnostics, "μ_t_negative", false)
        
        if has_negative_mu
            @warn "Generated material field with negative shear moduli - returning penalty" diagnostics
            diagnostics["intermediate_frac"] = 0.0
            diagnostics["severity"] = 0.0
            diagnostics["gray"] = 0.0
            return -1e3, 1e9, zeros(Float64, N_ELEM), diagnostics
        end
    end
    
    # Build stiffness matrices with new material field
    build_KE_fn(dh, mf, N_LOC, avg_mp_store)
    
    # Run topology optimization with error handling
    local X, c
    try
        X, c = topopt_run_fn(1)
    catch e
        if isa(e, DomainError)
            println("\n⚠️  DomainError during topology optimization - assigning penalty")
            
            # Log failure
            stacktrace_str = sprint(showerror, e, catch_backtrace())
            log_topology_failure(output_root, coeffs_dict, mf, string(e), stacktrace_str)
            
            diagnostics["intermediate_frac"] = 0.0
            diagnostics["severity"] = 0.0
            diagnostics["gray"] = 0.0
            
            return -1000.0, 1e6, zeros(N_ELEM), diagnostics
        else
            rethrow(e)
        end
    end
    
    # Compute adversarial metrics
    frac = compute_intermediary_fraction(X)
    severity = compute_intermediary_severity(X)
    gray = compute_gray_indicator(X)
    
    diagnostics["intermediate_frac"] = frac
    diagnostics["severity"] = severity
    diagnostics["gray"] = gray
    
    # Badness will be computed by optimizer with adaptive compliance reference
    # For now, return a basic combined metric
    badness = compute_combined_badness(
        X, c;
        w_frac=0.4,
        w_sev=0.4,
        w_gray=0.2,
        compliance_ref=c,
        compliance_penalty_threshold=10.0,
        stability_weight=0.15
    )
    
    return badness, c, X, diagnostics
end

"""
    create_objective_function(kl_modes_dict, properties, n_modes, mp, coords_elem, 
                             dh, mf, avg_mp_store, topopt_fn, build_KE_fn; 
                             validation_tolerance=10.0, output_root="output")

Create a closure that captures all necessary context for objective evaluation.
Returns a function with signature: `f(coeffs_mat) -> (badness, compliance, X, diagnostics)`
"""
function create_objective_function(kl_modes_dict::Dict, 
                                   properties::Tuple, 
                                   n_modes::Dict,
                                   mp,
                                   coords_elem,
                                   dh,
                                   mf,
                                   avg_mp_store,
                                   topopt_fn::Function,
                                   build_KE_fn::Function;
                                   validation_tolerance::Float64=10.0,
                                   output_root::String="output")
    
    return function(coeffs_mat::Matrix{Float64})
        return evaluate_objective(
            coeffs_mat, kl_modes_dict, properties, n_modes,
            mp, coords_elem, dh, mf, avg_mp_store,
            topopt_fn, build_KE_fn;
            validation_tolerance=validation_tolerance,
            output_root=output_root
        )
    end
end
