using JLD2

# Find most recent eigenmode file
output_dir = raw"C:\Users\judef\Desktop\COMMAS\SEM4\TopOpt\output"
cache_dir = joinpath(output_dir, "eigenmode_cache")

println("Checking eigenmode cache...")
if isdir(cache_dir)
    cache_files = readdir(cache_dir, join=true)
    jld2_files = filter(f -> endswith(f, ".jld2"), cache_files)
    
    if !isempty(jld2_files)
        latest = sort(jld2_files, by=f->stat(f).mtime, rev=true)[1]
        println("\nMost recent cache file: $latest")
        println("Modified: $(stat(latest).mtime)")
        
        data = JLD2.load(latest)
        kl_modes = data["kl_modes"]
        
        println("\n" * "="^60)
        println("EIGENMODE CONFIGURATIONS:")
        println("="^60)
        for (prop, kl) in kl_modes
            println("$prop:")
            println("  mode: $(kl.mode)")
            println("  mean_value: $(kl.mean_value)")
            println("  n_modes: $(length(kl.eigenvalues))")
            println("  eigenvalue range: [$(minimum(kl.eigenvalues)), $(maximum(kl.eigenvalues))]")
        end
        println("="^60)
    else
        println("No .jld2 files found in cache")
    end
else
    println("Cache directory doesn't exist")
end
