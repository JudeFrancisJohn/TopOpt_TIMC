"""
adversarial_optimizer.jl

Main optimization loop for adversarial coefficient search.
Uses CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for robust,
gradient-free optimization in high-dimensional coefficient space.
"""

using Random
using Statistics
using Plots

using Dates
using Printf
# Save history as JLD2
using JLD2
include("adversarial_utils.jl") 
# CMA-ES will be loaded from main script
# This module provides the optimization wrapper

"""
    AdversarialOptimizer

Configuration struct for adversarial optimization.
"""
mutable struct AdversarialOptimizer
    # Dimensions
    n_modes::Dict{Symbol, Int}
    properties::Tuple
    max_modes::Int
    n_props::Int
    
    # Optimization settings
    max_iterations::Int
    population_size::Int
    initial_sigma::Float64
    
    # Objective function weights
    w_frac::Float64
    w_severity::Float64
    w_gray::Float64
    
    # Bounds for coefficients
    coeff_lower_bound::Float64
    coeff_upper_bound::Float64
    
    # Logging
    save_path::String
    log_file::String
    save_every::Int  # Save checkpoint every N iterations
    
    # History tracking (metrics only, no coefficient bloat)
    history::Dict{String, Vector{Any}}
    
    # Best result tracking (minimal storage)
    best_badness::Float64
    best_iteration::Int
    best_coeffs::Union{Nothing, Matrix{Float64}}
    best_density_field::Union{Nothing, Vector{Float64}}
    
    function AdversarialOptimizer(n_modes::Dict{Symbol,Int}, properties::Tuple;
                                 max_iterations::Int=100,
                                 population_size::Int=0,  # 0 = auto
                                 initial_sigma::Float64=0.5,
                                 w_frac::Float64=0.4,
                                 w_severity::Float64=0.4,
                                 w_gray::Float64=0.2,
                                 coeff_bounds::Tuple{Float64,Float64}=(-3.0, 3.0),
                                 save_path::String="",
                                 save_every::Int=10)
        
        max_modes = maximum(values(n_modes))
        n_props = length(properties)
        
        # Auto-determine population size if not specified
        n_params = sum(values(n_modes))
        if population_size == 0
            population_size = 4 + floor(Int, 3 * log(n_params))
        end
        
        # Create output directory
        if isempty(save_path)
            dt_str = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
            save_path = joinpath("output", "adversarial_$(dt_str)")
        end
        mkpath(save_path)
        
        log_file = joinpath(save_path, "optimization_log.txt")
        
        # Initialize history tracking (ESSENTIAL METRICS ONLY - no coefficient bloat)
        history = Dict{String, Vector{Any}}(
            "iteration" => Int[],
            "badness" => Float64[],
            "compliance" => Float64[],
            "intermediate_frac" => Float64[],
            "severity" => Float64[],
            "gray_indicator" => Float64[],
            "timestamp" => String[]
        )
        
        new(n_modes, properties, max_modes, n_props,
            max_iterations, population_size, initial_sigma,
            w_frac, w_severity, w_gray,
            coeff_bounds[1], coeff_bounds[2],
            save_path, log_file, save_every,
            history,
            -Inf, 0, nothing, nothing)  # Initialize best tracking (including density field)
    end
end

"""
    initialize_coefficients(opt::AdversarialOptimizer; seed=nothing)

Initialize coefficient matrix with random values.
Uses standard normal distribution, optionally with seed.

# Arguments
- `opt`: AdversarialOptimizer instance
- `seed`: Random seed for reproducibility

# Returns
Initial coefficient matrix (max_modes × n_props)
"""
function initialize_coefficients(opt::AdversarialOptimizer; seed=nothing)
    if !isnothing(seed)
        Random.seed!(seed)
    end
    
    coeffs_mat = zeros(Float64, opt.max_modes, opt.n_props)
    
    for (j, prop) in enumerate(opt.properties)
        n = opt.n_modes[prop]
        # Initialize with small random values
        coeffs_mat[1:n, j] .= randn(n) * 0.1
    end
    
    return coeffs_mat
end

"""
    apply_bounds!(x::Vector{Float64}, lower::Float64, upper::Float64)

Apply box constraints to coefficient vector in-place.
Clips values outside [lower, upper] bounds.
"""
function apply_bounds!(x::Vector{Float64}, lower::Float64, upper::Float64)
    clamp!(x, lower, upper)
end

"""
    log_iteration(opt::AdversarialOptimizer, iteration::Int, 
                  badness::Float64, compliance::Float64,
                  frac::Float64, severity::Float64, gray::Float64,
                  coeffs_mat::Matrix{Float64}, X::Vector{Float64})

Log optimization iteration to history and file.
Tracks best result including density field.
"""
function log_iteration(opt::AdversarialOptimizer, iteration::Int, 
                      badness::Float64, compliance::Float64,
                      frac::Float64, severity::Float64, gray::Float64,
                      coeffs_mat::Matrix{Float64}, X::Vector{Float64};
                      best_so_far::Float64=badness)
    
    # Add to history (METRICS ONLY - no coefficient bloat)
    push!(opt.history["iteration"], iteration)
    push!(opt.history["badness"], badness)
    push!(opt.history["compliance"], compliance)
    push!(opt.history["intermediate_frac"], frac)
    push!(opt.history["severity"], severity)
    push!(opt.history["gray_indicator"], gray)
    push!(opt.history["timestamp"], string(Dates.now()))
    
    # Track best result (only store one set of coefficients + density field)
    if badness > opt.best_badness
        opt.best_badness = badness
        opt.best_iteration = iteration
        opt.best_coeffs = copy(coeffs_mat)
        opt.best_density_field = copy(X)  # Save best density field
    end
    
    # Write to log file
    open(opt.log_file, "a") do io
        if iteration == 1
            # Write header
            println(io, "# Adversarial Optimization Log")
            println(io, "# Started: $(Dates.now())")
            println(io, "# Max iterations: $(opt.max_iterations)")
            println(io, "# Population size: $(opt.population_size)")
            println(io, "# Number of parameters: $(sum(values(opt.n_modes)))")
            println(io, "# NOTE: 'Badness' shows current evaluation, 'Best' shows best-so-far (monotonic)")
            println(io, "#" ^ 100)
            @printf(io, "%-8s %-12s %-12s %-12s %-12s %-12s %-12s\n",
                   "Iter", "Badness", "Best", "Compliance", "Frac", "Severity", "Gray")
            println(io, "-" ^ 100)
        end
        
        @printf(io, "%-8d %-12.6f %-12.6f %-12.6e %-12.6f %-12.6f %-12.6f\n",
               iteration, badness, best_so_far, compliance, frac, severity, gray)
        flush(io)
    end
end

"""
    save_checkpoint(opt::AdversarialOptimizer, iteration::Int, 
                   coeffs_mat::Matrix{Float64}, X::Vector{Float64})

Save checkpoint - objective metrics only (no coefficients).
All iteration objectives are already logged in optimization_log.txt
"""
function save_checkpoint(opt::AdversarialOptimizer, iteration::Int, 
                        coeffs_mat::Matrix{Float64}, X::Vector{Float64})
    
    # NOTE: No checkpoint files needed!
    # All objective data is already in:
    #   - optimization_log.txt (text table)
    #   - optimization_history.jld2 (will be saved at end)
    
    println("  → Checkpoint marker at iteration $iteration (objectives in log)")
end

"""
    save_final_results(opt::AdversarialOptimizer, 
                      final_coeffs::Matrix{Float64},
                      final_badness::Float64)

Save final optimization results - ESSENTIALS ONLY (no bloat).
"""
function save_final_results(opt::AdversarialOptimizer, 
                           final_coeffs::Matrix{Float64},
                           final_badness::Float64)
    
    println("\n" * "=" ^ 100)
    println("OPTIMIZATION COMPLETED")
    println("=" ^ 100)
    
    # Use tracked best results
    best_idx = findfirst(==(opt.best_badness), opt.history["badness"])
    
    println("\nBest Result:")
    @printf("  Iteration:           %d\n", opt.best_iteration)
    @printf("  Badness:             %.6f\n", opt.best_badness)
    @printf("  Compliance:          %.6e\n", opt.history["compliance"][best_idx])
    @printf("  Intermediate Frac:   %.4f (%.1f%%)\n", 
            opt.history["intermediate_frac"][best_idx],
            opt.history["intermediate_frac"][best_idx] * 100)
    @printf("  Severity:            %.6f\n", opt.history["severity"][best_idx])
    @printf("  Gray Indicator:      %.6f\n", opt.history["gray_indicator"][best_idx])
    
    println("\nFinal Result:")
    @printf("  Badness:             %.6f\n", final_badness)
    
    # Save best coefficients (from tracked best)
    best_coeff_file = joinpath(opt.save_path, "best_coefficients.txt")
    export_coefficients_to_txt(opt.best_coeffs, opt.properties, opt.n_modes,
                               best_coeff_file; include_stats=true)
    
    # Save final coefficients
    final_coeff_file = joinpath(opt.save_path, "final_coefficients.txt")
    export_coefficients_to_txt(final_coeffs, opt.properties, opt.n_modes,
                               final_coeff_file; include_stats=true)
    
    # Save best density field separately
    density_file = joinpath(opt.save_path, "best_density_field.jld2")
    jldsave(density_file; 
            X=opt.best_density_field,
            iteration=opt.best_iteration,
            badness=opt.best_badness)
    
    # Save MINIMAL history (metrics only, no coefficient bloat)
    history_file = joinpath(opt.save_path, "optimization_history.jld2")
    jldsave(history_file;
            history=opt.history,                    # Metrics only (no coeffs stored)
            best_coeffs=opt.best_coeffs,            # Only ONE set of best coefficients
            final_coeffs=final_coeffs,              # Only ONE set of final coefficients
            best_badness=opt.best_badness,
            best_iteration=opt.best_iteration,
            final_badness=final_badness,
            n_modes=opt.n_modes,
            properties=opt.properties,
            settings=Dict(
                "max_iterations" => opt.max_iterations,
                "population_size" => opt.population_size,
                "initial_sigma" => opt.initial_sigma,
                "weights" => (opt.w_frac, opt.w_severity, opt.w_gray),
                "bounds" => (opt.coeff_lower_bound, opt.coeff_upper_bound)
            ))
    
    println("\nResults saved to: $(opt.save_path)")
    println("  - Best coefficients: best_coefficients.txt")
    println("  - Final coefficients: final_coefficients.txt")
    println("  - Best density field: best_density_field.jld2")
    println("  - Iteration objectives: optimization_log.txt (text table)")
    println("  - Complete data: optimization_history.jld2 (metrics + best/final coeffs)")
    println("=" ^ 100)
end

"""
    create_convergence_plot(opt::AdversarialOptimizer)

Create convergence plots for optimization metrics.
Saves plots to optimizer's save_path.
"""
function create_convergence_plot(opt::AdversarialOptimizer)
    try

        iters = opt.history["iteration"]
        
        # Create multi-panel plot
        p1 = plot(iters, opt.history["badness"], 
                 label="Badness", lw=2, marker=:circle,
                 xlabel="Iteration", ylabel="Badness",
                 title="Adversarial Objective")
        
        p2 = plot(iters, opt.history["compliance"],
                 label="Compliance", lw=2, marker=:square, color=:red,
                 xlabel="Iteration", ylabel="Compliance",
                 title="Structural Compliance", yaxis=:log10)
        
        p3 = plot(iters, opt.history["intermediate_frac"] .* 100,
                 label="Intermediate %", lw=2, marker=:diamond, color=:green,
                 xlabel="Iteration", ylabel="Percentage",
                 title="Intermediate Density Fraction")
        
        p4 = plot(iters, opt.history["gray_indicator"],
                 label="Gray Indicator", lw=2, marker=:star, color=:purple,
                 xlabel="Iteration", ylabel="GI",
                 title="Gray Indicator")
        
        combined = plot(p1, p2, p3, p4, layout=(2,2), size=(1200, 800))
        
        plot_file = joinpath(opt.save_path, "convergence.png")
        savefig(combined, plot_file)
        
        println("Convergence plot saved to: $plot_file")
        
    catch e
        @warn "Could not create convergence plot" exception=e
    end
end
