using LinearAlgebra, Printf,SparseArrays,StaticArrays
using Plots,Dates,Statistics, DelimitedFiles
using Ferrite
using TickTock, Parameters, Random
using IterativeSolvers
using WriteVTK
using Distributions
using Arpack

# Include dependencies
include("../utils/opt.jl")
include("../utils/FE_updated_stoch.jl")
include("../utils/stochastic_utils.jl")
include("../input/params_topopt.jl")
include("../input/params_mat.jl")
include("geom_BC.jl")

mp = MaterialParams(λ, μ_l, μ_t, alpha, beta, angle)

# NOTE: save_path must be defined by the caller!

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
        
        if loop % 10 == 0 || loop == 1
            println("    [TopOpt] Iteration $loop, change=$(round(change, digits=6))")
        end
        
        FE_Run!(ℂ, x)  # Pass design variables to FE analysis
        
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

        dc = check(nelx, nely, rmin, x, dc)
        dc = max.(dc, -abs.(dc) * 1e-12)
                
        x = OC(x, volfrac, dc)
        actual_volfrac = sum(x)/num_cells
        
        dt = Dates.format(Dates.now(), "yyyymmdd")
        angle_str = replace(string(mp.angle), "." => "p")
        iter_sfx = loop
        unique_name = "st_$(dt)_$(lx)_$(ly)_$(angle_str)_$(rmin)_$(penal)_$(iter_sfx)"
        
        # Use global save_path directly
        run_dir = save_path 
        
        # Export results
        full_vtu = joinpath(run_dir, string(unique_name) * ".vtu")
        export_vtk(u, dh, grid, cv_post, mp, ip, run_dir, unique_name; density = x)
        
        change = maximum(abs.(x .- xold))
        push!(vol_history, actual_volfrac)

        last_compliance = c
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
