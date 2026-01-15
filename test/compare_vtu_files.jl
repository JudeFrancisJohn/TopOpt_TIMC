"""
compare_vtu_files.jl

Compare material field data between original and reconstructed VTU files.
"""

using ReadVTK

function read_vtu_data(filename::String)
    println("\n📄 Reading: $filename")
    
    # Read the VTU file
    vtu = VTKFile(filename)
    
    # Get cell data (material fields)
    cell_data_dict = get_cell_data(vtu)
    println("  Available cell data fields: ", keys(cell_data_dict))
    
    data = Dict{String, Any}()
    for field_name in ["μ_l", "μ_t", "alpha", "beta", "lambda", "angle", "density"]
        if haskey(cell_data_dict, field_name)
            data[field_name] = cell_data_dict[field_name]
            vals = data[field_name]
            println("  $field_name: min=$(minimum(vals)), max=$(maximum(vals)), mean=$(sum(vals)/length(vals))")
        else
            println("  $field_name: NOT FOUND")
        end
    end
    
    return data
end

function compare_fields(original_file::String, reconstructed_file::String)
    println("="^80)
    println("COMPARING VTU FILES")
    println("="^80)
    
    orig_data = read_vtu_data(original_file)
    recon_data = read_vtu_data(reconstructed_file)
    
    println("\n" * "="^80)
    println("COMPARISON")
    println("="^80)
    
    for field_name in keys(orig_data)
        if haskey(recon_data, field_name)
            orig_vals = orig_data[field_name]
            recon_vals = recon_data[field_name]
            
            diff = orig_vals .- recon_vals
            max_diff = maximum(abs.(diff))
            mean_diff = sum(abs.(diff)) / length(diff)
            
            println("\n$field_name:")
            println("  Max absolute difference: $max_diff")
            println("  Mean absolute difference: $mean_diff")
            
            if max_diff > 1e-6
                println("  ⚠️  SIGNIFICANT DIFFERENCE DETECTED")
                # Show some examples
                large_diff_idx = findall(abs.(diff) .> mean_diff * 2)
                if length(large_diff_idx) > 0
                    println("  Examples of large differences:")
                    for i in large_diff_idx[1:min(5, end)]
                        println("    Element $i: original=$(orig_vals[i]), reconstructed=$(recon_vals[i]), diff=$(diff[i])")
                    end
                end
            else
                println("  ✅ Fields match")
            end
        else
            println("\n$field_name: MISSING in reconstructed file")
        end
    end
end

# Main execution
if abspath(PROGRAM_FILE) == @__FILE__
    run_dir = "output/adversarial_20260114_22"
    original = joinpath(run_dir, "best_iteration_0039.vtu")
    reconstructed = joinpath(run_dir, "best_iteration_0039_reconstructed.vtu")
    
    if !isfile(original)
        println("ERROR: Original file not found: $original")
        exit(1)
    end
    if !isfile(reconstructed)
        println("ERROR: Reconstructed file not found: $reconstructed")
        exit(1)
    end
    
    compare_fields(original, reconstructed)
end
