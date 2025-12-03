using LinearAlgebra, Printf,SparseArrays,StaticArrays
using Plots,Dates,Statistics, DelimitedFiles
using Ferrite
using TickTock, Parameters, Random
using IterativeSolvers
using WriteVTK
using Distributions
using Arpack

# TODO: avg_mp has to be passed in each element_routine
# TODO: find a way to store avg mp for each element

#  TODO : 
#  - Introduce stochastics to material parameters
# -  KL Expansion
# -  Done: Modify C in each element_routine
# -  Assemble this new stiffness matrix that changes in each MC run
# -   TO DO: COPY PASTE THE TOP OPT part and check if it still runs?
#======================================#

include("../utils/opt.jl")
include("../utils/FE_updated_stoch.jl")
include("../utils/stochastic_utils.jl")

include("../input/params_topopt.jl")
include("../input/params_mat.jl")
include("geom_BC.jl")
mp = MaterialParams(λ, μ_l, μ_t, alpha, beta, angle)

run_mode = "stochastic"
dt = Dates.format(Dates.now(), "yyyymmdd_HH")
run_name = "$(run_mode)_$(dt)"   # change this string to a custom run name if desired


save_root = joinpath(@__DIR__, "..", "output")
save_path = joinpath(save_root, run_name)
mkpath(save_path)

remove_files(save_path)
export_vtk(u_d, dh, grid, cv_post, mp, ip, save_path, 0)

cells = getcells(grid)
elements = [cell.nodes for cell in cells]
nodes = getnodes(grid)
nelem = size(elements)
nnodes = size(nodes)

#Geometry and element/node counts
dim = 2
nnodes_per_cell = length(cells[1].nodes)   # 4 for quadrilateral (nodes per element)
nelem = length(cells)
C = zeros(3,3)
# store a Vec{2,Float64} for each element/node: shape (nelem, nnodes_per_cell)
coords_elem = Array{Vec{2,Float64}}(undef, nelem, nnodes_per_cell)
for ei in 1:nelem
    v = getcoordinates(grid, ei)   # Vector{Vec{2,Float64}} of length nnodes_per_cell
    @assert length(v) == nnodes_per_cell
    for ni in 1:nnodes_per_cell
        coords_elem[ei, ni] = v[ni]
    end
end
# coords_elem[1,:]


ndofs_per_cell_local = ndofs_per_cell(dh)
Bmat = Array{Float64}(undef, 3, ndofs_per_cell_local)
ϵ   = Vector{Float64}(undef, 3)
σ   = similar(ϵ)
ℂ   = Array{Float64}(undef, 3, 3)
KE_store = Vector{Matrix{Float64}}(undef, nelem)

num_cells = length(CellIterator(dh))
if num_cells == (nelx*nely)
    println("Pre-check: Grid size matches: $(num_cells) cells = $(nelx)×$(nely)")
end

n = ndofs_per_cell(dh)
ke = Array{Float64}(undef, n, n)
ue = Vector{Float64}(undef, n)
ge = similar(ue)
global_dofs = Vector{Int}(undef, n)
λf = 0
ℂ = Array{Float64}(undef, 3, 3)
num_cells = length(CellIterator(dh))
ℂ_store = Vector{Matrix{Float64}}(undef, num_cells)
nnodes_loc = Int(ndofs_per_cell(dh) ÷ dim)
avg_mp_store = Vector{MaterialParams}(undef, num_cells)

#TopologyOpt params initialization
x = ones(num_cells)  #START WITH FULL DENSITY for stable initial solve
change = 1.0
loop = 0
vol_history = Float64[]

function topopt_run(run_i)
    x = ones(num_cells) #START WITH FULL DENSITY for stable initial solve
    change = 1.0
    loop = 0
    vol_history = Float64[]
    last_compliance = 0
    
    println("    [TopOpt] Starting optimization loop...")
    while change > 0.01 && loop < 1000
        loop += 1
        xold = copy(x)
        
        # Progress indicator every 10 iterations
        if loop % 10 == 0 || loop == 1
            println("    [TopOpt] Iteration $loop, change=$(round(change, digits=6))")
        end
        
        # println("\n" * "-"^80)
        # println("ITERATION $loop")
        # println("-"^80)
        # println("Step 1: Running FE Analysis with current design...")
        # println("  Density stats: min=$(round(minimum(x), digits=4)), max=$(round(maximum(x), digits=4)), mean=$(round(mean(x), digits=4))")
        # println("  Penalized stiffness multiplier range: [$(round(minimum(x)^penal, digits=6)), $(round(maximum(x)^penal, digits=6))]")
        #---------------------------#
        FE_Run!(ℂ, x)  # Pass design variables to FE analysis
        #----------------------------#
        # println("✓ FE Analysis complete")
        
        # # --- Compute compliance and sensitivities ---
        # println("\nStep 2: Computing compliance and sensitivities...")
        c = 0.0
        dc = zeros(num_cells)
        for (cell_index, cell) in enumerate(CellIterator(dh))
            local global_dofs = celldofs(cell)
            local ue = u[global_dofs]
            local ke = KE_store[cell_index]  

            x_cell = x[cell_index]
            c += x_cell^penal * (ue' * ke * ue)
            dc[cell_index] = -penal * x_cell^(penal-1) * (ue' * ke * ue)
        end
        # println("  Compliance (objective): $(round(c, digits=6))")
        # println("  Sensitivity range: [$(round(minimum(dc), digits=6)), $(round(maximum(dc), digits=6))]")

        # println("\nStep 3: Applying sensitivity filter...")
        dc = check(nelx, nely, rmin, x, dc)
        dc = max.(dc, -abs.(dc) * 1e-12)
        # println("  Filtered sensitivity range: [$(round(minimum(dc), digits=6)), $(round(maximum(dc), digits=6))]")
                
        # # DESIGN UPDATE
        # println("\nStep 4: Updating design variables (OC method)...")
        x = OC(x, volfrac, dc)
        actual_volfrac = sum(x)/num_cells
        # println("  Density range: [$(round(minimum(x), digits=4)), $(round(maximum(x), digits=4))]")
        # println("  Volume fraction: $(round(actual_volfrac, digits=4)) (target: $(volfrac))")
        
        # # --- Export all results to Paraview in a single VTK file ---
        # println("\nStep 5: Exporting results to Paraview...")
    dt = Dates.format(Dates.now(), "yyyymmdd")
    angle_str = replace(string(mp.angle), "." => "p")
    # make iteration suffix explicit and zero-padded for consistent sorting
    iter_sfx = loop
    unique_name = "st_$(dt)_$(lx)_$(ly)_$(angle_str)_$(rmin)_$(penal)_$(iter_sfx)"
        # create a per-run subfolder so each stochastic run gets its own directory
        run_dir = joinpath(save_path, "run_$(run_i)")
        mkpath(run_dir)

        # Export results to VTK using centralized helper which projects quad data -> nodes
        # and writes point/cell fields. Pass current density `x` as cell data.
        full_vtu = joinpath(run_dir, string(unique_name) * ".vtu")
        export_vtk(u, dh, grid, cv_post, mp, ip, run_dir, unique_name; density = x)
        # println("  Saved: ", full_vtu)
        
        change = maximum(abs.(x .- xold))
        push!(vol_history, actual_volfrac)

        last_compliance = c
        # println("\n" * "="^80)
        # println("ITERATION $loop SUMMARY:")
        # println("  Objective (Compliance):  $(round(last_compliance, digits=4))")
        # # println("  Volume Fraction:         $(round(actual_volfrac, digits=4))")
        # println("  Change:                  $(round(change, digits=6))")
        # # println("="^80)


    end

    println("\n" * "="^80)
    println("║" * " "^23 * "TOPOLOGY OPTIMIZATION COMPLETE" * " "^23 * "║")
    println("="^80)
    println("Final Results:")
    println("  • Total iterations: $loop")
    println("  • Final objective: $(round(last_compliance, digits=4))")
    println("  • Final volume fraction: $(round(sum(x)/num_cells, digits=4))")
    println("  • Final change: $(round(change, digits=6))")
    println("="^80)
    return x, last_compliance
end
 

function multiple_runs(nruns=3)

    for i in range(1,nruns)
        print("RUN  - $(i)")
        # Now with stochastics, each realization generates new set of mf parameters
        # Use iteration index as seed for reproducibility
        fields = KL_realization(mp, coords_elem;
                        σs = Dict(:μ_l => 0.8 * mp.μ_l,   # 80% of mean μ_l
                            :μ_t => 0.1 * mp.μ_t,          # 10% of mean μ_t
                            :α   => 0.8 * mp.alpha,        # 80% of mean α
                            :β   => 0.8 * mp.beta),        # 80% of mean β
                        Lc=0.01, N_modes=80, use_centroids=false,
                        make_sparse=true, kernel=:exponential, mode=:lognormal,
                        seed=1000 + i)  # Reproducible seed: 1001, 1002, 1003, ...

        mf = build_material_field(fields; use_centroids=false, eltype_out=Float32)


        #pre-allocate stiffness matrices for elements once
        build_KEStore!(dh, mf, nnodes_loc, avg_mp_store)

        # Reset for topology optimization
        global u = zeros(ndofs(dh))

        # println("\n" * "="^80)
        # println("║" * " "^25 * "TOPOLOGY OPTIMIZATION START" * " "^26 * "║")
        # println("="^80)
        # println("Parameters:")
        # println("  • Grid: $(nelx)×$(nely) elements ($(num_cells) total)")
        # println("  • Target volume fraction: $(volfrac)")
        # println("  • Penalization (SIMP): $(penal)")
        # println("  • Filter radius: $(rmin)")
        # println("  • Convergence criterion: change < 0.01")
        # println("="^80 * "\n")


        X,c = topopt_run(i)
    end
end

#multiple_runs(3)