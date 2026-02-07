using JLD2

file = raw"C:\Users\judef\Desktop\COMMAS\SEM4\TopOpt\output\adversarial_20260125_2038\eigenmodes.jld2"
data = JLD2.load(file)
kl_modes = data["kl_modes"]

println("Checking eigenmode modes:")
for (prop_sym, kl) in kl_modes
    println("  $prop_sym: mode = $(kl.mode), mean = $(kl.mean_value)")
end
