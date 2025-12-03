using JLD2, Plots

# Load best density field
density_data = load("")
X_best = density_data["X"]          # Density field (1200 elements)
best_iter = density_data["iteration"]  # Which iteration produced this
best_bad = density_data["badness"]     # Badness score

# Visualize (60×20 mesh)
heatmap(reshape(X_best, (60, 20))', yflip=true, c=:grays, clim=(0,1))

# Check intermediate density stats
intermediates = X_best[(X_best .>= 0.1) .& (X_best .<= 0.9)]
println("Intermediate fraction: ", length(intermediates)/length(X_best) * 100, "%")