using LinearAlgebra, Printf,SparseArrays,StaticArrays
using Plots,Dates,Statistics, DelimitedFiles
using Ferrite, FerriteMeshParser
using TickTock, Parameters, Random
using IterativeSolvers
using WriteVTK
using Distributions
using Arpack

#to do: avg_mp has to be passed in each element_routine
# find a way to store avg mp for each element
# =========================
# Goal : Introduce stochastics to material parameters
#  -  KL Expansion
# -  Done: Modify C in each element_routine
# -  Assemble this new stiffness matrix that changes in each MC run
# -   TO DO: COPY PASTE THE TOP OPT part and check if it still runs?
#======================================#

include("input.jl")
include("opt.jl")
include("FE_updated_stoch.jl")
include("stochastic_utils.jl")

mp = MaterialParams(λ, μ_l, μ_t, alpha, beta, angle)
corners = [
    Vec{2}((0.0, 0.0)),
    Vec{2}((lx, 0.0)),
    Vec{2}((lx, ly)),
    Vec{2}((0.0, ly)),
]

grid = generate_grid(Quadrilateral, (nelx, nely), corners)

# Robust node/face selection: use isapprox with small atol so whole edges are selected
addnodeset!(grid, "left_edge", x -> isapprox(x[1], 0.0; atol=1e-8));
addnodeset!(grid, "right_bottom_node", x -> isapprox(x[1], lx; atol=1e-8) && isapprox(x[2], 0.0; atol=1e-8));
# bottom-left corner (single node) to pin vertical DOF and remove remaining rigid body motion
addfaceset!(grid, "topmid_face", x -> (isapprox(x[2], ly; atol=1e-8) && abs(x[1]) <= 0.5))

dim = 2;
ip_g = Lagrange{dim, RefCube, 1}();
ip = Lagrange{dim, RefCube, 1}(); # other option would be 8 noded Serendipity elements
qpo = 2;
qr = QuadratureRule{dim, RefCube}(qpo);
cv = CellScalarValues(Float64,qr, ip, ip_g);

fqr = QuadratureRule{dim-1,RefCube}(qpo);
fv = FaceVectorValues(fqr, ip, ip_g);

# one gauss point integration for postprocessing
nqp_post = 1;
qr_post = QuadratureRule{dim, RefCube}(nqp_post);
cv_post = CellScalarValues(Float64,qr_post, ip, ip_g);

#ΓN = getfaceset(grid, "right_face"); # Neumann Boundary

dh = DofHandler(grid);
push!(dh, :u, 2, ip); # Displacement vector
close!(dh);

ch = ConstraintHandler(dh);

∂Ωl = getnodeset(dh.grid, "left_edge");
# Fix horizontal displacement on the entire left edge (roller in x)
add!(ch, Ferrite.Dirichlet(:u, ∂Ωl, (x,t) -> 0.0000, 1));


# Right edge: apply a vertical roller support (fix vertical DOF only)
∂Ωr = getnodeset(dh.grid, "right_bottom_node");
add!(ch, Ferrite.Dirichlet(:u, ∂Ωr, (x,t) -> 0.0000, 2));

"""∂Ωr = getnodeset(dh.grid, "right_edge");
dbcrv = Dirichlet(:u, ∂Ωr, (x,t) -> 0.2*t, 2); # Vertical Displacement
add!(ch, dbcrv);"""

close!(ch);
Ferrite.update!(ch, 0.0);  # Explicitly use Ferrite's update! to avoid ambiguity


# dof vector
uₙ = Vector{Float64}(undef,ndofs(dh));
fill!(uₙ,zero(eltype(uₙ)));

u = Vector{Float64}(undef,ndofs(dh));
fill!(u,zero(eltype(u)));
u .= uₙ;

save_path = mkpath("./FEOutputs/$(basename(@__DIR__))/stochastic");
# clean_savepath(save_path)
exportresults(uₙ, dh, grid, cv_post, mp, ip, save_path, 0)


cells = getcells(grid)
elements = [cell.nodes for cell in cells]
nodes = getnodes(grid)
nelem = size(elements)
nnodes = size(nodes)

dim = 2
nloc = length(cells[1].nodes)   # 4 for quadrilateral
nelem = length(cells)
C = zeros(3,3)
# store a Vec{2,Float64} for each element/node: shape (nelem, nloc)
coords_elem = Array{Vec{2,Float64}}(undef, nelem, nloc)
for ei in 1:nelem
    v = getcoordinates(grid, ei)   # Vector{Vec{2,Float64}} of length nloc
    @assert length(v) == nloc
    for ni in 1:nloc
        coords_elem[ei, ni] = v[ni]
    end
end
# coords_elem[1,:]


nloc = ndofs_per_cell(dh)
Bmat = Array{Float64}(undef, 3, nloc)
ϵ   = Vector{Float64}(undef, 3)
σ   = similar(ϵ)
ℂ   = Array{Float64}(undef, 3, 3)

# Now with stochastics, each realization generates new set of mf parameters

fields = KL_realization(mp, coords_elem;
                        σs = Dict(:μ_l => 0.5 * mp.μ_l,   # 50% of mean μ_l
                                :μ_t => 0.5 * mp.μ_t,
                                :α   => 0.5 * mp.alpha,
                                :β   => 0.5 * mp.beta),
                        Lc=0.05, N_modes=80, use_centroids=false,
                        make_sparse=true, kernel=:exponential, mode=:lognormal)

mf = build_material_field(fields; use_centroids=false, eltype_out=Float32)

function element_routine!(ke, ge, cell, dh, cv, fv, mp, ue, ℂ,λf)
    # Reinitialize cell and reset element matrices
    reinit!(cv, cell)
    fill!(ke, 0.0)
    fill!(ge, 0.0)

    nloc = ndofs_per_cell(dh)
    Bmat = Array{Float64}(undef, 3, nloc)
    ϵ   = Vector{Float64}(undef, 3)
    σ   = similar(ϵ)

    # ----------------- Volume integral -----------------
    for qp in 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, qp)
        fill!(Bmat, 0.0)

        for i in 1:getnbasefunctions(cv)
            gN = shape_gradient(cv, qp, i)
            Bmat[1, 2*i-1] = gN[1]
            Bmat[2, 2*i]   = gN[2]
            Bmat[3, 2*i-1] = gN[2]
            Bmat[3, 2*i]   = gN[1]
        end

        ϵ .= Bmat * ue
        # use the material params passed into the element routine (avg per-element mp)
        trans_iso_stress!(σ, ϵ, mp)

        ge .-= Bmat' * (dΩ * σ)        # NEGATIVE: -f_internal (will add f_ext later)
        ke .+= Bmat' * (dΩ * ℂ) * Bmat # tangent

    end

    
    # ----------------- Surface integral (Neumann BC) -----------------
    nen = getnbasefunctions(cv)
    ndim  = 2
    ndofs = length(ge)
    traction = Vec(0.0,1.0)*λf

    @inbounds for face in 1:nfaces(cell)
            if (cellid(cell), face) ∈ getfaceset(grid, "topmid_face")
                reinit!(fv, cell, face)
                for qp in 1:getnquadpoints(fv)
                    dΓ = getdetJdV(fv, qp)
                    for i in 1:ndofs
                        δu = shape_value(fv, qp, i)
                        ge[i] += (δu ⋅ traction) * dΓ;  # POSITIVE: +f_external
                    end
                end
            end
        end

    return ke
    
end

# --- VTK Export Functions ---

"""
Export displacement (u) as cell data to Paraview
"""
function export_u_to_vtk(filename, grid, dh, u)
    # For cell data, we need to average nodal displacements per element
    num_cells = length(CellIterator(dh))
    avg_u_x = zeros(num_cells)
    avg_u_y = zeros(num_cells)
    
    for (cell_index, cell) in enumerate(CellIterator(dh))
        global_dofs = celldofs(cell)
        ue = u[global_dofs]
        # Average x and y displacements for this cell
        n_nodes = length(global_dofs) ÷ 2
        avg_u_x[cell_index] = sum(ue[1:2:end]) / n_nodes
        avg_u_y[cell_index] = sum(ue[2:2:end]) / n_nodes
    end
    
    vtk = vtk_grid(filename, grid)
    vtk_cell_data(vtk, avg_u_x, "u_x")
    vtk_cell_data(vtk, avg_u_y, "u_y")
    vtk_save(vtk)
end

"""
Export strain (epsilon) as cell data to Paraview
"""
function export_epsilon_to_vtk(filename, grid, dh, cv_post, mp, u)
    num_cells = length(CellIterator(dh))
    avg_epsilon_xx = zeros(num_cells)
    avg_epsilon_yy = zeros(num_cells)
    avg_epsilon_xy = zeros(num_cells)
    
    for (cell_index, cell) in enumerate(CellIterator(dh))
        reinit!(cv_post, cell)
        global_dofs = celldofs(cell)
        ue = u[global_dofs]
        nqp = getnquadpoints(cv_post)
        epsilon_sum = zeros(3)
        Bmat = Array{Float64}(undef, 3, ndofs_per_cell(dh))
        
        for qp in 1:nqp
            fill!(Bmat, 0.0)
            for i in 1:getnbasefunctions(cv_post)
                gN = shape_gradient(cv_post, qp, i)
                Bmat[1, 2*i-1] = gN[1]
                Bmat[2, 2*i]   = gN[2]
                Bmat[3, 2*i-1] = gN[2]
                Bmat[3, 2*i]   = gN[1]
            end
            epsilon_sum .+= Bmat * ue
        end
        
        avg_epsilon_xx[cell_index] = epsilon_sum[1] / nqp
        avg_epsilon_yy[cell_index] = epsilon_sum[2] / nqp
        avg_epsilon_xy[cell_index] = epsilon_sum[3] / nqp
    end
    
    vtk = vtk_grid(filename, grid)
    vtk_cell_data(vtk, avg_epsilon_xx, "epsilon_xx")
    vtk_cell_data(vtk, avg_epsilon_yy, "epsilon_yy")
    vtk_cell_data(vtk, avg_epsilon_xy, "epsilon_xy")
    vtk_save(vtk)
end

"""
Export stress (sigma) as cell data to Paraview
"""
function export_sigma_to_vtk(filename, grid, dh, cv_post, mp, u)
    num_cells = length(CellIterator(dh))
    avg_sigma_xx = zeros(num_cells)
    avg_sigma_yy = zeros(num_cells)
    avg_sigma_xy = zeros(num_cells)
    
    for (cell_index, cell) in enumerate(CellIterator(dh))
        reinit!(cv_post, cell)
        global_dofs = celldofs(cell)
        ue = u[global_dofs]
        nqp = getnquadpoints(cv_post)
        sigma_sum = zeros(3)
        Bmat = Array{Float64}(undef, 3, ndofs_per_cell(dh))
        epsilon = Vector{Float64}(undef, 3)
        sigma = Vector{Float64}(undef, 3)
        
        for qp in 1:nqp
            fill!(Bmat, 0.0)
            for i in 1:getnbasefunctions(cv_post)
                gN = shape_gradient(cv_post, qp, i)
                Bmat[1, 2*i-1] = gN[1]
                Bmat[2, 2*i]   = gN[2]
                Bmat[3, 2*i-1] = gN[2]
                Bmat[3, 2*i]   = gN[1]
            end
            epsilon .= Bmat * ue
                trans_iso_stress!(sigma, epsilon, avg_mp_store[cell_index])
            sigma_sum .+= sigma
        end
        
        avg_sigma_xx[cell_index] = sigma_sum[1] / nqp
        avg_sigma_yy[cell_index] = sigma_sum[2] / nqp
        avg_sigma_xy[cell_index] = sigma_sum[3] / nqp
    end
    
    vtk = vtk_grid(filename, grid)
    vtk_cell_data(vtk, avg_sigma_xx, "sigma_xx")
    vtk_cell_data(vtk, avg_sigma_yy, "sigma_yy")
    vtk_cell_data(vtk, avg_sigma_xy, "sigma_xy")
    vtk_save(vtk)
end

function assemble_global!(K, g, dh, cv, fv, mp, u, ℂ, x, λf)
    # Number of DOFs per element
    n = ndofs_per_cell(dh)
    ke = Array{Float64}(undef, n, n)
    ue = Vector{Float64}(undef, n)
    ge = similar(ue)
    global_dofs = Vector{Int}(undef, n)

    # Start assembling global K and g
    assembler = start_assemble(K, g)

    # -------------------- Volume (internal) contributions --------------------
    for (cell_index,cell) in enumerate(CellIterator(dh))
        global_dofs .= celldofs(cell)   # element DOFs
        ue .= u[global_dofs]            # element displacement vector
        
        # For SIMP: The actual stiffness is K_SIMP = x^p * K
        # So the internal force should be: f_int = x^p * K * u
        # We can compute this directly using the precomputed KE
        x_simp = x[cell_index]^penal
        
        # Compute internal force: f_int = x^p * K * u (element level)
        f_int = x_simp * (KE_store[cell_index] * ue)
        
        # Get external forces from element routine (only surface traction)
        fill!(ge, 0.0)
        # Add surface traction if this element has it
        nen = getnbasefunctions(cv)
        ndim  = 2
        ndofs = length(ge)
        traction = Vec(0.0,1.0)*λf
        @inbounds for face in 1:nfaces(cell)
            if (cellid(cell), face) ∈ getfaceset(grid, "topmid_face")
                reinit!(fv, cell, face)
                for qp in 1:getnquadpoints(fv)
                    dΓ = getdetJdV(fv, qp)
                    for i in 1:ndofs
                        δu = shape_value(fv, qp, i)
                        ge[i] += (δu ⋅ traction) * dΓ
                    end
                end
            end
        end
        
        # Residual: g = f_ext - f_int (with SIMP-penalized internal force)
        ge .-= f_int
        
        # Assemble: penalized stiffness and consistent residual
        assemble!(assembler, global_dofs, x_simp * KE_store[cell_index], ge)
    end

    #println("Global stiffness matrix assembly completed")
    return K, ndofs(dh)
end


function NonlinearSolve(dh, cv, fv, ch, mp, uₙ, ℂ,x, λf)

    _ndofs = ndofs(dh);
    u = zeros(_ndofs);
    Δu = zeros(_ndofs);
    ΔΔu = zeros(_ndofs);
    apply!(uₙ, ch);
    # Create sparse matrix and residual vector
    K = create_sparsity_pattern(dh); #Create the sparsity pattern corresponding to the degree of freedom numbering in the DofHandler. Return a SparseMatrixCSC with stored values in the correct places.
    g = zeros(_ndofs);


    newton_itr = -1;
    NEWTON_TOL = 1e-10;
    normref = 1.0;

    
    while true; newton_itr += 1

        u .= uₙ .+ Δu;      # updating displacements
        assemble_global!(K, g, dh, cv, fv, mp, u, ℂ,x,λf);

        if newton_itr==0
            normref = norm(g[Ferrite.free_dofs(ch)]);
        end

        apply_zero!(K, g, ch);
        normg = norm(g[Ferrite.free_dofs(ch)]);
        normrel = normg/normref;
        if isnan(normrel)
            error("NaN in residuum")
        end
    # printstyled("relative residual norm after $newton_itr iterations is $normrel \n";color=:yellow)

        if normrel < NEWTON_TOL || normg < NEWTON_TOL
            # printstyled("Converged after $newton_itr iterations with relative residual norm $normrel \n";color=:green)
            break
        elseif newton_itr > 10
            error("Reached maximum Newton iterations: $newton_itr, aborting")
        elseif normg>1e15
            error("Residuum blow up to $normg !!, aborting")
        end

        # Newton-Raphson: K * ΔΔu = g where g = f_ext - f_int
        # This gives the displacement correction
        ΔΔu = K \ g;
        apply_zero!(ΔΔu, ch)
        Δu .+= ΔΔu;  # Update displacement increment


    end

    return u

end;


function FE_Run!(ℂ, x_fe)
    #---
    tick()

    nsteps=100;
    Δt = 1e-2;
    local_t = 0.0  # Use local time for load stepping
    u_local = zeros(ndofs(dh))  # Start from zero displacement each FE run
    
    for i=1:nsteps
        local_t += Δt;
        Ferrite.update!(ch, local_t);  # Explicitly use Ferrite's update! to avoid ambiguity
        λf = i/nsteps
        # u .= Solve(dh, cv, fv, ch, mp, uₙ);
        u_local = NonlinearSolve(dh, cv, fv, ch, mp, u_local, ℂ, x_fe, λf);
        #exportresults(u_local, dh, grid, cv_post, mp, ip, save_path, i);      
    end
    
    global u = u_local  # Update global displacement
   
    tock()
end


dim = 2
nloc = length(cells[1].nodes)   # 4 for quadrilateral
nelem = length(cells)

# store a Vec{2,Float64} for each element/node: shape (nelem, nloc)
coords_elem = Array{Vec{2,Float64}}(undef, nelem, nloc)
for ei in 1:nelem
    v = getcoordinates(grid, ei)   # Vector{Vec{2,Float64}} of length nloc
    @assert length(v) == nloc
    for ni in 1:nloc
        coords_elem[ei, ni] = v[ni]
    end
end

coords_elem[1,:]
    

num_cells = length(CellIterator(dh))
if num_cells == (nelx*nely)
    println("✓ Grid size matches: $(num_cells) cells = $(nelx)×$(nely)")
end
x = ones(num_cells)  # START WITH FULL DENSITY for stable initial solve
change = 1.0
loop = 0
vol_history = Float64[]

KE_store = Vector{Matrix{Float64}}(undef, num_cells)
#material_routine!(ℂ, σ, ϵ, mp)



# Number of DOFs per element
n = ndofs_per_cell(dh)
ke = Array{Float64}(undef, n, n)
ue = Vector{Float64}(undef, n)
ge = similar(ue)
global_dofs = Vector{Int}(undef, n)
λf = 0

ℂ = Array{Float64}(undef, 3, 3)
#trans_iso_tangent!(ℂ,mp)

mf = build_material_field(fields; use_centroids=false)

num_cells = length(CellIterator(dh))
ℂ_store = Vector{Matrix{Float64}}(undef, num_cells)
nnodes_loc = Int(ndofs_per_cell(dh) ÷ dim)
avg_mp_store = Vector{MaterialParams}(undef, num_cells)

for (cell_index, cell) in enumerate(CellIterator(dh))

    μ_l_sum = 0.0; μ_t_sum = 0.0; α_sum = 0.0; β_sum = 0.0; λ_sum = 0.0; ang_sum = 0.0

    for ni in 1:nnodes_loc
        m = get_material(mf, cell_index, ni)
        μ_l_sum += m.μ_l
        μ_t_sum += m.μ_t
        α_sum   += m.α
        β_sum   += m.β
        λ_sum   += m.λ
        ang_sum += m.angle
    end

    μ_l_avg = μ_l_sum / nnodes_loc
    μ_t_avg = μ_t_sum / nnodes_loc
    α_avg   = α_sum   / nnodes_loc
    β_avg   = β_sum   / nnodes_loc
    λ_avg   = λ_sum   / nnodes_loc
    ang_avg = ang_sum / nnodes_loc

    avg_mp = MaterialParams(λ_avg, μ_l_avg, μ_t_avg, α_avg, β_avg, ang_avg)
    avg_mp_store[cell_index] = avg_mp

    C = zeros(3,3)
    trans_iso_tangent!(C, avg_mp)    
    ℂ_store[cell_index] = C
    fill!(ke, 0.0)
    fill!(ge, 0.0)
    # use per-element averaged material params when computing element stiffness
    KE_store[cell_index] = element_routine!(ke, ge, cell, dh, cv, fv, avg_mp, zeros(n), ℂ_store[cell_index], 0.0)
end


# TEST: First run FEA without SIMP to verify it works
println("\n" * "="^80)
println("║" * " "^28 * "FEA VERIFICATION TEST" * " "^29 * "║")
println("="^80)
println("Testing FEA with uniform density (x = 1.0 everywhere)")
x_test = ones(num_cells)
FE_Run!(ℂ, x_test)
println("✓ FEA test completed successfully!")
println("="^80 * "\n")



# Reset for topology optimization
global u = zeros(ndofs(dh))

println("\n" * "="^80)
println("║" * " "^25 * "TOPOLOGY OPTIMIZATION START" * " "^26 * "║")
println("="^80)
println("Parameters:")
println("  • Grid: $(nelx)×$(nely) elements ($(num_cells) total)")
println("  • Target volume fraction: $(volfrac)")
println("  • Penalization (SIMP): $(penal)")
println("  • Filter radius: $(rmin)")
println("  • Convergence criterion: change < 0.01")
println("="^80 * "\n")

while change > 0.01 && loop < 1000
    global loop +=1
    xold = copy(x)
    
    println("\n" * "-"^80)
    println("ITERATION $loop")
    println("-"^80)
    println("Step 1: Running FE Analysis with current design...")
    println("  Density stats: min=$(round(minimum(x), digits=4)), max=$(round(maximum(x), digits=4)), mean=$(round(mean(x), digits=4))")
    println("  Penalized stiffness multiplier range: [$(round(minimum(x)^penal, digits=6)), $(round(maximum(x)^penal, digits=6))]")
    FE_Run!(ℂ, x)  # Pass design variables to FE analysis
    println("✓ FE Analysis complete")
    
    # --- Compute compliance and sensitivities ---
    println("\nStep 2: Computing compliance and sensitivities...")
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
    println("  Compliance (objective): $(round(c, digits=6))")
    println("  Sensitivity range: [$(round(minimum(dc), digits=6)), $(round(maximum(dc), digits=6))]")

    println("\nStep 3: Applying sensitivity filter...")
    dc = check(nelx, nely, rmin, x, dc)
    dc = max.(dc, -abs.(dc) * 1e-12)
    println("  Filtered sensitivity range: [$(round(minimum(dc), digits=6)), $(round(maximum(dc), digits=6))]")
            
    # DESIGN UPDATE
    println("\nStep 4: Updating design variables (OC method)...")
    global x = OC(x, volfrac, dc)
    actual_volfrac = sum(x)/num_cells
    println("  Density range: [$(round(minimum(x), digits=4)), $(round(maximum(x), digits=4))]")
    println("  Volume fraction: $(round(actual_volfrac, digits=4)) (target: $(volfrac))")
    
    # --- Export all results to Paraview in a single VTK file ---
    println("\nStep 5: Exporting results to Paraview...")
    dt = Dates.format(Dates.now(), "yyyymmdd")
    unique_name = string("st_",dt, "_",lx,"_",ly,"_", mp.angle, "_", rmin, "_", penal,"_","TR_",loop)
    fullpath = joinpath(save_path, unique_name)
    
    # Compute all cell data
    avg_stress = zeros(num_cells)
    avg_u_x = zeros(num_cells)
    avg_u_y = zeros(num_cells)
    avg_epsilon_xx = zeros(num_cells)
    avg_epsilon_yy = zeros(num_cells)
    avg_epsilon_xy = zeros(num_cells)
    avg_sigma_xx = zeros(num_cells)
    avg_sigma_yy = zeros(num_cells)
    avg_sigma_xy = zeros(num_cells)
    
    for (cell_index, cell) in enumerate(CellIterator(dh))
        reinit!(cv_post, cell)
        global_dofs = celldofs(cell)
        ue = u[global_dofs]
        nqp = getnquadpoints(cv_post)
        
        # Average displacements
        n_nodes = length(global_dofs) ÷ 2
        avg_u_x[cell_index] = sum(ue[1:2:end]) / n_nodes
        avg_u_y[cell_index] = sum(ue[2:2:end]) / n_nodes
        
        # Compute stress and strain at quadrature points
        σ_qp = Vector{Float64}(undef, 3)
        σ_sum = 0.0
        epsilon_sum = zeros(3)
        sigma_sum = zeros(3)
        Bmat_local = Array{Float64}(undef, 3, ndofs_per_cell(dh))
        epsilon = Vector{Float64}(undef, 3)
        sigma = Vector{Float64}(undef, 3)
        
        for qp in 1:nqp
            fill!(Bmat_local, 0.0)
            for i in 1:getnbasefunctions(cv_post)
                gN = shape_gradient(cv_post, qp, i)
                Bmat_local[1, 2*i-1] = gN[1]
                Bmat_local[2, 2*i]   = gN[2]
                Bmat_local[3, 2*i-1] = gN[2]
                Bmat_local[3, 2*i]   = gN[1]
            end
            epsilon .= Bmat_local * ue
            # use stored averaged material params for this cell
            trans_iso_stress!(sigma, epsilon, avg_mp_store[cell_index])
            
            # von Mises stress
            σ_vm = sqrt(sigma[1]^2 + sigma[2]^2 - sigma[1]*sigma[2] + 3*sigma[3]^2)
            σ_sum += σ_vm
            
            # Accumulate strain and stress
            epsilon_sum .+= epsilon
            sigma_sum .+= sigma
        end
        
        # Apply SIMP penalization to stress (void elements should have near-zero stress)
        x_cell = x[cell_index]
        avg_stress[cell_index] = (σ_sum / nqp) * x_cell^penal
        avg_epsilon_xx[cell_index] = (epsilon_sum[1] / nqp) * x_cell^penal
        avg_epsilon_yy[cell_index] = (epsilon_sum[2] / nqp) * x_cell^penal
        avg_epsilon_xy[cell_index] = (epsilon_sum[3] / nqp) * x_cell^penal
        avg_sigma_xx[cell_index] = (sigma_sum[1] / nqp) * x_cell^penal
        avg_sigma_yy[cell_index] = (sigma_sum[2] / nqp) * x_cell^penal
        avg_sigma_xy[cell_index] = (sigma_sum[3] / nqp) * x_cell^penal
    end
    
    # Write all data to a single VTK file
    vtk = vtk_grid(fullpath, grid)
    vtk_cell_data(vtk, x, "density")
    vtk_cell_data(vtk, avg_stress, "von_mises_stress")
    vtk_cell_data(vtk, avg_u_x, "u_x")
    vtk_cell_data(vtk, avg_u_y, "u_y")
    vtk_cell_data(vtk, avg_epsilon_xx, "epsilon_xx")
    vtk_cell_data(vtk, avg_epsilon_yy, "epsilon_yy")
    vtk_cell_data(vtk, avg_epsilon_xy, "epsilon_xy")
    vtk_cell_data(vtk, avg_sigma_xx, "sigma_xx")
    vtk_cell_data(vtk, avg_sigma_yy, "sigma_yy")
    vtk_cell_data(vtk, avg_sigma_xy, "sigma_xy")
    vtk_save(vtk)
    println("  Saved: ", unique_name, ".vtu")
    
    global change = maximum(abs.(x .- xold))
    push!(vol_history, actual_volfrac)

    global last_compliance = c
    println("\n" * "="^80)
    println("ITERATION $loop SUMMARY:")
    println("  Objective (Compliance):  $(round(last_compliance, digits=4))")
    println("  Volume Fraction:         $(round(actual_volfrac, digits=4))")
    println("  Change:                  $(round(change, digits=6))")
    println("="^80)


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






