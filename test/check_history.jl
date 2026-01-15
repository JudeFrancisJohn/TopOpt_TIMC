using JLD2

jldopen("output/adversarial_20260114_22/optimization_history.jld2", "r") do f
    history = read(f, "history")
    println("History type: ", typeof(history))
    println("History keys: ", keys(history))
    
    # Check structure
    for (key, val) in history
        println("\nKey: $key")
        println("  Type: ", typeof(val))
        println("  Length: ", length(val))
        if length(val) > 0
            println("  First entry type: ", typeof(val[1]))
            if isa(val[1], Dict) || isa(val[1], Pair)
                println("  First entry keys: ", keys(val[1]))
            end
        end
    end
    
    # Try to access iteration data
    if haskey(history, "iteration")
        iterations = history["iteration"]
        badness_vals = history["badness"]
        println("\n\nIterations: ", iterations)
        println("Badness values: ", badness_vals)
        
        best_idx = argmax(badness_vals)
        println("\nBest iteration index: $best_idx")
        println("Best iteration number: ", iterations[best_idx])
        println("Best badness: ", badness_vals[best_idx])
    end
end
