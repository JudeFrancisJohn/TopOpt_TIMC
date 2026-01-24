"""
adversarial_logging.jl

Logging, export, and tracking utilities for adversarial optimization.
Handles VTU file management, metadata persistence, and best iteration tracking.
"""

using Dates
using Printf

# ============================================================================
# BEST ITERATION TRACKING
# ============================================================================

"""
    BestVTUTracker

Tracks the best iteration found during optimization and manages VTU file cleanup.
Only one VTU file is kept at a time to save disk space.
"""
mutable struct BestVTUTracker
    badness::Float64
    compliance::Float64
    iter::Int
    vtu_path::Union{String, Nothing}
end

"""
    init_best_tracker()

Initialize a new BestVTUTracker with default values.
"""
function init_best_tracker()
    return BestVTUTracker(-Inf, Inf, 0, nothing)
end

"""
    delete_previous_vtu!(tracker::BestVTUTracker)

Delete the previous best VTU file to save disk space.
Called when a new best iteration is found.
"""
function delete_previous_vtu!(tracker::BestVTUTracker)
    if tracker.vtu_path !== nothing && isfile(tracker.vtu_path)
        try
            rm(tracker.vtu_path; force=true)
            println("    🗑️  Deleted previous best VTU: $(tracker.vtu_path)")
        catch e
            @warn "Failed to delete previous best VTU" path=tracker.vtu_path error=e
        end
    end
end

"""
    save_best_vtu!(tracker, X, iter_id; save_dir, export_fn)

Save VTU file for the current best iteration.

# Arguments
- `tracker::BestVTUTracker`: Tracker to update
- `X::AbstractVector`: Density field to export
- `iter_id::Int`: Current iteration number
- `save_dir::String`: Directory to save VTU file
- `export_fn::Function`: Function to call for VTU export (e.g., export_vtk)
"""
function save_best_vtu!(tracker::BestVTUTracker, X::AbstractVector{<:Real}, iter_id::Int, 
                        save_dir::AbstractString, export_fn::Function)
    mkpath(save_dir)
    fname = joinpath(save_dir, @sprintf("best_iteration_%04d.vtu", iter_id))
    
    try
        # Call the provided export function
        # Assumes signature: export_fn(save_dir, filename_prefix; density=X)
        fname_prefix = @sprintf("best_iteration_%04d", iter_id)
        export_fn(save_dir, fname_prefix; density=X)
        
        tracker.vtu_path = fname
        println("    💾 Saved best VTU: $(fname)")
    catch e
        @warn "Failed to export best VTU" iter=iter_id error=e
    end
end

"""
    persist_best_metadata(save_dir, tracker)

Write metadata file for the best iteration found so far.
"""
function persist_best_metadata(save_dir::AbstractString, tracker::BestVTUTracker)
    meta_file = joinpath(save_dir, "best_metadata.txt")
    open(meta_file, "w") do io
        println(io, "Best iteration metadata")
        println(io, "timestamp: ", Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))
        println(io, @sprintf("badness: %.6f", tracker.badness))
        println(io, @sprintf("compliance: %.6f", tracker.compliance))
        println(io, "iteration: ", tracker.iter)
        println(io, "vtu_path: ", tracker.vtu_path === nothing ? "" : tracker.vtu_path)
    end
    println("    📝 Wrote metadata: $(meta_file)")
end

# ============================================================================
# FAILURE DIAGNOSTICS
# ============================================================================

"""
    log_topology_failure(output_root, coeffs_dict, mf, error_msg, stacktrace_str)

Log detailed diagnostics when topology optimization fails.
Saves material field statistics, coefficients, and error details.
"""
function log_topology_failure(output_root::String, coeffs_dict::Dict, mf, error_msg::String, 
                               stacktrace_str::String; density_field=nothing)
    failure_file = joinpath(output_root, "failed_topopt_runs.txt")
    open(failure_file, "a") do io
        println(io, "\n" * "="^80)
        println(io, "Failed TopOpt Run - $(Dates.now())")
        println(io, "="^80)
        println(io, "Error: $error_msg")
        
        println(io, "\nKL Coefficients:")
        for (prop, coeffs) in coeffs_dict
            println(io, "  $prop: $(coeffs)")
        end
        
        println(io, "\nMaterial Field Statistics:")
        for prop_sym in fieldnames(typeof(mf))
            val = getfield(mf, prop_sym)
            if isa(val, AbstractArray)
                println(io, "  $prop_sym: mean=$(mean(val)), std=$(std(val)), min=$(minimum(val)), max=$(maximum(val))")
            end
        end
        
        if density_field !== nothing
            println(io, "\nDensity Field x:")
            println(io, "  mean=$(mean(density_field)), std=$(std(density_field)), min=$(minimum(density_field)), max=$(maximum(density_field))")
            println(io, "  First 20 elements: $(density_field[1:min(20, length(density_field))])")
        else
            println(io, "\nDensity Field x: Not yet initialized")
        end
        
        println(io, "\nStacktrace:")
        println(io, stacktrace_str)
        println(io, "="^80)
    end
    println("   → Diagnostics saved to: $failure_file")
end

# ============================================================================
# OPTIMIZATION PROGRESS PRINTING
# ============================================================================

"""
    print_optimization_summary(iter, badness, compliance, frac, severity, gray)

Print a formatted summary of the current optimization iteration.
"""
function print_optimization_summary(iter::Int, badness::Float64, compliance::Float64, 
                                     frac::Float64, severity::Float64, gray::Float64)
    println("\n" * "="^80)
    println("ITERATION $iter SUMMARY")
    println("="^80)
    println(@sprintf("  Badness (objective): %.6f", badness))
    println(@sprintf("  Compliance:          %.3e", compliance))
    println(@sprintf("  Intermediate frac:   %.4f", frac))
    println(@sprintf("  Severity:            %.4f", severity))
    println(@sprintf("  Gray indicator:      %.4f", gray))
end

"""
    print_best_update(badness, iter, best_so_far)

Print notification when a new best solution is found.
"""
function print_best_update(badness::Float64, iter::Int, best_so_far::Float64)
    println("  🎯 NEW BEST! Badness=$(round(badness, digits=6)) at iteration $iter")
    println("     Previous best: $(round(best_so_far, digits=6))")
end

"""
    print_optimization_header(save_path, properties, n_modes, max_iterations)

Print header information at the start of optimization.
"""
function print_optimization_header(save_path::String, properties::Tuple, n_modes::Dict, 
                                    max_iterations::Int)
    println("\n" * "="^80)
    println("ADVERSARIAL COEFFICIENT OPTIMIZATION")
    println("="^80)
    println("Output directory: $(save_path)")
    println("\nOptimization Settings:")
    println("  Max iterations:      $(max_iterations)")
    println("  Total parameters:    $(sum(values(n_modes)))")
    println("  Properties optimized: $(join(properties, ", "))")
end
