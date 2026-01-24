"""
io_manager.jl

Handles all I/O operations for adversarial optimization:
- Console output (progress, summaries, updates)
- Checkpoint saving
- VTU file management
- History logging
- Result exporting

Following DRY principle: One class for all I/O concerns.
"""

using Dates
using Printf
using JLD2
using Statistics

include("adversarial_logging.jl")

# ============================================================================
# IO MANAGER CLASS
# ============================================================================

"""
    IOManager

Centralized I/O handler for optimization process.
Manages all printing, saving, and export operations.

# Fields
- `save_path::String`: Root directory for outputs
- `save_every::Int`: Checkpoint frequency
- `best_tracker::BestVTUTracker`: Tracks best iteration VTU
- `history::Dict`: Optimization history log
- `verbose::Bool`: Enable detailed console output
"""
mutable struct IOManager
    save_path::String
    save_every::Int
    best_tracker::BestVTUTracker
    history::Dict{String, Vector{Any}}
    verbose::Bool
    
    function IOManager(save_path::String; save_every::Int=10, verbose::Bool=true)
        history = Dict{String, Vector{Any}}(
            "iteration" => Int[],
            "badness" => Float64[],
            "compliance" => Float64[],
            "intermediate_frac" => Float64[],
            "severity" => Float64[],
            "gray_indicator" => Float64[],
            "timestamp" => String[]
        )
        
        best_tracker = init_best_tracker()
        
        new(save_path, save_every, best_tracker, history, verbose)
    end
end

# ============================================================================
# INITIALIZATION & HEADER
# ============================================================================

"""
    print_header(io::IOManager, properties, n_modes, max_iterations, optimizer_type)

Print optimization header at start.
"""
function print_header(io::IOManager, properties::Tuple, n_modes::Dict, 
                      max_iterations::Int, optimizer_type::String)
    if !io.verbose
        return
    end
    
    println("\nStarting $(optimizer_type) optimization...")
    println("="^80)
    println("Output directory: $(io.save_path)")
    println("\nOptimization Settings:")
    println("  Max iterations:      $(max_iterations)")
    println("  Total parameters:    $(sum(values(n_modes)))")
    println("  Properties optimized: $(join(properties, ", "))")
    println("  Checkpoint frequency: Every $(io.save_every) iterations")
    println("="^80)
end

# ============================================================================
# ITERATION LOGGING
# ============================================================================

"""
    log_iteration(io::IOManager, iter, badness, compliance, diagnostics)

Log iteration data to history.
"""
function log_iteration(io::IOManager, iter::Int, badness::Float64, 
                       compliance::Float64, diagnostics::Dict)
    push!(io.history["iteration"], iter)
    push!(io.history["badness"], badness)
    push!(io.history["compliance"], compliance)
    push!(io.history["intermediate_frac"], diagnostics["intermediate_frac"])
    push!(io.history["severity"], diagnostics["severity"])
    push!(io.history["gray_indicator"], diagnostics["gray"])
    push!(io.history["timestamp"], string(Dates.now()))
end

"""
    print_iteration_summary(io::IOManager, iter, badness, compliance, 
                           diagnostics, best_so_far)

Print periodic iteration summary.
"""
function print_iteration_summary(io::IOManager, iter::Int, badness::Float64, 
                                 compliance::Float64, diagnostics::Dict, 
                                 best_so_far::Float64)
    if !io.verbose
        return
    end
    
    frac = diagnostics["intermediate_frac"]
    severity = diagnostics["severity"]
    gray = diagnostics["gray"]
    
    println("\n" * "="^80)
    println("ITERATION $iter SUMMARY")
    println("="^80)
    println(@sprintf("  Badness (objective): %.6f", badness))
    println(@sprintf("  Compliance:          %.3e", compliance))
    println(@sprintf("  Intermediate frac:   %.4f", frac))
    println(@sprintf("  Severity:            %.4f", severity))
    println(@sprintf("  Gray indicator:      %.4f", gray))
    println("    Best so far: $(round(best_so_far, digits=6))")
end

# ============================================================================
# BEST SOLUTION TRACKING
# ============================================================================

"""
    handle_new_best(io::IOManager, badness, compliance, iter, X, 
                    coeffs_mat, export_fn)

Handle discovery of new best solution.
Updates tracker, manages VTU files, saves metadata.
"""
function handle_new_best(io::IOManager, badness::Float64, compliance::Float64, 
                         iter::Int, X::Vector{Float64}, coeffs_mat::Matrix{Float64},
                         export_fn::Function)
    old_best = io.best_tracker.badness
    
    # Update tracker
    io.best_tracker.badness = badness
    io.best_tracker.compliance = compliance
    io.best_tracker.iter = iter
    
    # Print notification
    if io.verbose
        println("  🎯 NEW BEST! Badness=$(round(badness, digits=6)) at iteration $iter")
        println("     Previous best: $(round(old_best, digits=6))")
    end
    
    # Delete previous VTU
    delete_previous_vtu!(io.best_tracker)
    
    # Save new VTU
    function export_wrapper(save_dir, filename_prefix; density=nothing)
        export_fn(save_dir, filename_prefix, density)
    end
    
    save_best_vtu!(io.best_tracker, X, iter, io.save_path, export_wrapper)
    
    # Persist metadata
    persist_best_metadata(io.save_path, io.best_tracker)
end

# ============================================================================
# CHECKPOINT MANAGEMENT
# ============================================================================

"""
    save_checkpoint(io::IOManager, iter, coeffs_mat, X)

Save optimization checkpoint to disk.
"""
function save_checkpoint(io::IOManager, iter::Int, coeffs_mat::Matrix{Float64}, 
                         X::Vector{Float64})
    checkpoint_file = joinpath(io.save_path, "checkpoint_iter_$(iter).jld2")
    jldsave(checkpoint_file; coefficients=coeffs_mat, density_field=X, iteration=iter)
    
    if io.verbose
        println("    💾 Saved checkpoint: $(checkpoint_file)")
    end
end

"""
    should_save_checkpoint(io::IOManager, iter)

Determine if checkpoint should be saved at this iteration.
"""
function should_save_checkpoint(io::IOManager, iter::Int)
    return (iter % io.save_every == 0)
end

# ============================================================================
# FINAL RESULTS
# ============================================================================

"""
    save_final_results(io::IOManager, coeffs_mat, badness)

Save final optimization results (coefficients, history).
"""
function save_final_results(io::IOManager, coeffs_mat::Matrix{Float64}, badness::Float64; properties=nothing, n_modes=nothing)
    # Save coefficients as text
    coeffs_file = joinpath(io.save_path, "best_coefficients.txt")
    open(coeffs_file, "w") do file
        println(file, "Best Coefficients (badness = $(badness))")
        println(file, "Shape: $(size(coeffs_mat))")
        println(file, "\nMatrix:")
        for i in 1:size(coeffs_mat, 1)
            println(file, join(coeffs_mat[i, :], ", "))
        end
    end
    
    # Save history as JLD2 with best_coeffs key for reconstruct_best_vtu.jl
    history_file = joinpath(io.save_path, "optimization_history.jld2")
    if properties !== nothing && n_modes !== nothing
        jldsave(history_file; 
                history=io.history, 
                best_coeffs=coeffs_mat,  # Key expected by reconstruct_best_vtu.jl
                properties=collect(properties),
                n_modes=n_modes)
    else
        jldsave(history_file; 
                history=io.history,
                best_coeffs=coeffs_mat)  # Key expected by reconstruct_best_vtu.jl
    end
    
    if io.verbose
        println("\n✅ Final results saved:")
        println("   Coefficients: $(coeffs_file)")
        println("   History: $(history_file)")
    end
end

"""
    print_completion_summary(io::IOManager, best_badness)

Print final completion summary.
"""
function print_completion_summary(io::IOManager, best_badness::Float64)
    if !io.verbose
        return
    end
    
    println("\n" * "="^80)
    println("OPTIMIZATION COMPLETE!")
    println("="^80)
    println("\nBest badness: $(round(best_badness, digits=6))")
    println("\nResults saved to: $(io.save_path)")
    println("\nTo reconstruct material fields from best coefficients:")
    println("  1. Load coefficients from: best_coefficients.txt")
    println("  2. Load eigenmodes from: eigenmodes.jld2")
    println("  3. Use `sample_KL_field()` with pre-computed eigenmodes")
    println("  4. Build MaterialField and run TopOpt")
    println("\n✅ Done!")
end

"""
    create_convergence_plot(io::IOManager)

Create convergence plots for optimization metrics.
Requires Plots.jl to be available.
"""
function create_convergence_plot(io::IOManager)
    try
        # Lazy load Plots to avoid dependency if not used
        @eval import Plots: plot, savefig
        
        iters = io.history["iteration"]
        
        if isempty(iters)
            @warn "No iteration history to plot"
            return
        end
        
        # Create multi-panel plot
        p1 = Plots.plot(iters, io.history["badness"], 
             label="Badness", lw=2, marker=:circle,
             xlabel="Iteration", ylabel="Badness",
             title="Adversarial Objective", legend=:best)
        
        p2 = Plots.plot(iters, io.history["compliance"],
             label="Compliance", lw=2, marker=:square, color=:red,
             xlabel="Iteration", ylabel="Compliance",
             title="Structural Compliance", yaxis=:log10, legend=:best)
        
        p3 = Plots.plot(iters, io.history["intermediate_frac"] .* 100,
             label="Intermediate %", lw=2, marker=:diamond, color=:green,
             xlabel="Iteration", ylabel="Percentage",
             title="Intermediate Density Fraction", legend=:best)
        
        p4 = Plots.plot(iters, io.history["gray_indicator"],
             label="Gray Indicator", lw=2, marker=:star, color=:purple,
             xlabel="Iteration", ylabel="GI",
             title="Gray Indicator", legend=:best)
        
        combined = Plots.plot(p1, p2, p3, p4, layout=(2,2), size=(1200, 800))
        
        plot_file = joinpath(io.save_path, "convergence.png")
        Plots.savefig(combined, plot_file)
        
        if io.verbose
            println("📊 Convergence plot saved to: $plot_file")
        end
        
        return plot_file
        
    catch e
        if isa(e, ArgumentError) && occursin("Plots", string(e))
            @warn "Could not create convergence plot: Plots.jl not available. Install with: using Pkg; Pkg.add(\"Plots\")"
        else
            @warn "Could not create convergence plot" exception=(e, catch_backtrace())
        end
        return nothing
    end
end

# ============================================================================
# PROGRESS INDICATORS
# ============================================================================

"""
    should_print_summary(iter, frequency=10)

Determine if iteration summary should be printed.
"""
function should_print_summary(iter::Int, frequency::Int=10)
    return (iter % frequency == 0) || (iter == 1)
end
