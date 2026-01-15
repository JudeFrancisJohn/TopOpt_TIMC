"""
inspect_saved_data.jl

Diagnostic script to understand the structure of saved data in adversarial runs.
Inspects eigenmodes.jld2 and optimization_history.jld2 to see exactly what's stored.

Usage:
    julia --project=. test/inspect_saved_data.jl output/adversarial_20260114_22
"""

using JLD2
using Printf

function inspect_run(run_dir::AbstractString)
    println("="^80)
    println("INSPECTING RUN DIRECTORY: $run_dir")
    println("="^80)
    
    # ===== Inspect eigenmodes.jld2 =====
    eigen_file = joinpath(run_dir, "eigenmodes.jld2")
    if isfile(eigen_file)
        println("\n📁 EIGENMODES FILE: $eigen_file")
        println("-"^80)
        jldopen(eigen_file, "r") do f
            println("Keys in file: ", keys(f))
            
            kl_modes = read(f, "kl_modes")
            n_modes = read(f, "n_modes")
            
            println("\n🔑 kl_modes type: ", typeof(kl_modes))
            println("   Properties (keys): ", keys(kl_modes))
            
            for (prop, modes) in kl_modes
                println("\n   Property: $prop")
                println("     Type: ", typeof(modes))
                println("     Fields: ", fieldnames(typeof(modes)))
                if hasproperty(modes, :eigenvalues)
                    println("     Eigenvalues length: ", length(modes.eigenvalues))
                    println("     First 3 eigenvalues: ", modes.eigenvalues[1:min(3, end)])
                end
                if hasproperty(modes, :eigenvectors)
                    println("     Eigenvectors size: ", size(modes.eigenvectors))
                end
            end
            
            println("\n🔢 n_modes: ", n_modes)
            
            if haskey(f, "σs")
                σs = read(f, "σs")
                println("\n📊 σs (standard deviations): ", σs)
            end
        end
    else
        println("\n❌ eigenmodes.jld2 NOT FOUND")
    end
    
    # ===== Inspect optimization_history.jld2 =====
    history_file = joinpath(run_dir, "optimization_history.jld2")
    if isfile(history_file)
        println("\n\n📁 OPTIMIZATION HISTORY FILE: $history_file")
        println("-"^80)
        jldopen(history_file, "r") do f
            println("Keys in file: ", keys(f))
            
            # Read best_coeffs
            best_coeffs = read(f, "best_coeffs")
            println("\n🎯 best_coeffs:")
            println("   Type: ", typeof(best_coeffs))
            println("   Size: ", size(best_coeffs))
            println("   First few values: ", best_coeffs[1:min(10, length(best_coeffs))])
            
            # Read properties
            properties = read(f, "properties")
            println("\n🏷️  properties:")
            println("   Type: ", typeof(properties))
            println("   Value: ", properties)
            
            # Read n_modes
            if haskey(f, "n_modes")
                n_modes_hist = read(f, "n_modes")
                println("\n🔢 n_modes: ", n_modes_hist)
            end
            
            # Read iteration history if available
            if haskey(f, "iteration_history")
                history = read(f, "iteration_history")
                println("\n📈 iteration_history:")
                println("   Type: ", typeof(history))
                println("   Length: ", length(history))
                if length(history) > 0
                    println("\n   First entry:")
                    first_entry = history[1]
                    for (k, v) in first_entry
                        println("     $k: $v")
                    end
                    
                    println("\n   Last entry:")
                    last_entry = history[end]
                    for (k, v) in last_entry
                        println("     $k: $v")
                    end
                end
            end
            
            # Read best_iteration metadata
            if haskey(f, "best_iteration")
                println("\n⭐ best_iteration: ", read(f, "best_iteration"))
            end
            if haskey(f, "best_badness")
                println("   best_badness: ", read(f, "best_badness"))
            end
        end
    else
        println("\n❌ optimization_history.jld2 NOT FOUND")
    end
    
    println("\n" * "="^80)
end

# Main execution
if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 1
        println("Usage: julia --project=. test/inspect_saved_data.jl <RUN_DIR>")
        exit(1)
    end
    
    run_dir = ARGS[1]
    inspect_run(run_dir)
end
