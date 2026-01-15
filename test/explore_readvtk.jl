using ReadVTK

vtu = VTKFile("output/adversarial_20260114_22/best_iteration_0039.vtu")
println("Type: ", typeof(vtu))
println("Fieldnames: ", fieldnames(typeof(vtu)))

# Try to access data
try
    data = get_cell_data(vtu)
    println("Cell data keys: ", keys(data))
catch e
    println("get_cell_data error: ", e)
end

try
    data = get_data(vtu)
    println("Data keys: ", keys(data))
catch e
    println("get_data error: ", e)
end

# Check what methods are available
println("\nMethods for VTKFile:")
println(methods(VTKFile))
