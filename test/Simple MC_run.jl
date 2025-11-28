using Random
using Statistics
using Dates
using Printf


println("Loading TopOpt driver...")
include( "../src/COPY_stochastic_modified_v2_MC.jl")
include("../input/params_MCMC.jl")

const OUTPUT_ROOT = normpath(joinpath(@__DIR__, "..", "output"))
global save_root = OUTPUT_ROOT
dt_str = Dates.format(Dates.now(), "yyyymmdd_HH")
global save_path = joinpath(save_root, "mcmc_chain_$(dt_str)")
chain_file = normpath(joinpath(save_root, "mcmc_chain_$(dt_str).txt"))
println("Output file: $(chain_file)")
mkpath(dirname(chain_file))

