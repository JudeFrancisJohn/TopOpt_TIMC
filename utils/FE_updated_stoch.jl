using Logging
mutable struct MaterialField{T}
    μ_l::Array{T,2}    # nelem × nloc (or nelem × 1 if centroids)
    μ_t::Array{T,2}
    α::Array{T,2}
    β::Array{T,2}
    λ::Array{T,2}
    angle::Array{T,2}
    use_centroids::Bool
end

struct MaterialParams
    λ::Float64
    μ_l::Float64
    μ_t::Float64
    alpha::Float64
    beta::Float64
    angle::Float64
end

# infer nelem,nloc from fields dict (works with either matrices or centroid vectors)
function _infer_dims(fields::Dict{Symbol,Any})
    for v in values(fields)
        if isa(v, AbstractMatrix)
            return size(v)
        elseif isa(v, AbstractVector)
            return (length(v), 1)
        end
    end
    error("Cannot infer dimensions from fields dict")
end

# convert value to nelem×nloc matrix of desired eltype
function _to_matrix(v, nelem::Int, nloc::Int, eltype_out::Type, use_centroids::Bool)
    if isa(v, AbstractMatrix)
        return convert(Array{eltype_out,2}, v)
    elseif isa(v, AbstractVector)
        if use_centroids || nloc == 1
            return reshape(convert(Vector{eltype_out}, v), nelem, 1)
        else
            error("Expected 2D array for per-node values when use_centroids=false")
        end
    else
        # scalar -> broadcast
        return fill(eltype_out(v), nelem, nloc)
    end
end

# create MaterialField from KL fields Dict
function build_material_field(fields::Dict{Symbol,Any}; use_centroids::Bool=false, eltype_out::Type=Float32)
    nelem, nloc = _infer_dims(fields)
    μ_l_m  = _to_matrix(fields[:μ_l],  nelem, nloc, eltype_out, use_centroids)
    μ_t_m  = _to_matrix(fields[:μ_t],  nelem, nloc, eltype_out, use_centroids)
    α_m    = _to_matrix(fields[:α],    nelem, nloc, eltype_out, use_centroids)
    β_m    = _to_matrix(fields[:β],    nelem, nloc, eltype_out, use_centroids)
    λ_m    = _to_matrix(fields[:λ],    nelem, nloc, eltype_out, use_centroids)
    ang_m  = _to_matrix(fields[:angle],nelem, nloc, eltype_out, use_centroids)
    return MaterialField(μ_l_m, μ_t_m, α_m, β_m, λ_m, ang_m, use_centroids)
end

# fast accessor for use inside element loops
@inline function get_material(mf::MaterialField, ei::Int, ni::Int=1)
    return (
        μ_l = mf.μ_l[ei,ni],
        μ_t = mf.μ_t[ei,ni],
        α   = mf.α[ei,ni],
        β   = mf.β[ei,ni],
        λ   = mf.λ[ei,ni],
        angle = mf.angle[ei,ni],
    )
end

function build_KEStore!(dh,mf,nnodes_loc,avg_mp_store)
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
        tangent_transiso!(C, avg_mp)    
        ℂ_store[cell_index] = C
        fill!(ke, 0.0)
        fill!(ge, 0.0)
        # use per-element averaged material params when computing element stiffness
        KE_store[cell_index] = element_routine!(ke, ge, cell, dh, cv, fv, avg_mp, zeros(n), ℂ_store[cell_index], 0.0)
    end
end

@with_kw struct NeumannpointBoundaryinfo
    ΓNlist::Vector{Set};
    tractionlist::Vector{Vector{Float64}};
end

@views function tangent_transiso!(ℂ,mp)

    fill!(ℂ, zero(eltype(ℂ)));

    voigt_index = [[1,1],[2,2],[1,2]];
    δ = Matrix(1.0I, 2, 2);
    M = similar(δ);

    rot_angle = deg2rad(mp.angle);
    a = [cos(rot_angle),sin(rot_angle)];

    for i=1:2
        for j=1:2
            M[i,j] = a[i]*a[j];
        end
    end

    ℭ = Array{Float64,4}(undef,2,2,2,2);        #Creates a 4D array (2x2x2x2) to store the components of the tangent tensor
    II = similar(ℭ);
    IIa = similar(ℭ);
    for i=1:2
        for j=1:2
            for k=1:2
                for l=1:2
                    II[i,j,k,l] = (1/2)*(δ[i,k] * δ[j,l] + δ[i,l] * δ[j,k]);
                    IIa[i,j,k,l] = (1/2)*(M[i,k]*δ[j,l] + M[i,l]*δ[j,k] + M[j,l]*δ[i,k] + M[j,k]*δ[i,l]);
                    ℭ[i,j,k,l] = (mp.λ * δ[i,j] * δ[k,l] + 2.0 * mp.μ_t * II[i,j,k,l] + mp.alpha*(δ[i,j]*M[k,l] + M[i,j]*δ[k,l])
                                    + 2.0*(mp.μ_l -mp.μ_t)*IIa[i,j,k,l] + mp.beta*M[i,j]*M[k,l]);
                end
            end
        end
    end

    for i=1:3
        k = voigt_index[i][1]
        l = voigt_index[i][2]
        for j=1:3
            p = voigt_index[j][1]
            q = voigt_index[j][2]
            ℂ[i,j] = ℭ[k,l,p,q];         # Converting the Tangent moduli tensor to voigt notation
        end
    end


end

@views function stress_transiso!(σ,ϵ,mp)
    fill!(σ, zero(eltype(σ)));

    voigt_index = [[1,1],[2,2],[1,2]];
    rot_angle = deg2rad(mp.angle);
    a = [cos(rot_angle),sin(rot_angle)];

    ε = [ϵ[1] ϵ[3]/2;ϵ[3]/2 ϵ[2]];
    M = similar(ε);
    σmat = similar(ε);
    δ = Matrix(1.0I, 2, 2);

    for i=1:2
        for j=1:2
            M[i,j] = a[i]*a[j];
        end
    end

    I1 = tr(ε);
    I4 = tr(ε*M);
    σmat .= mp.λ*I1*δ + 2*mp.μ_t*ε + mp.alpha*( I4*δ + I1*M) + 2.0*(mp.μ_l -mp.μ_t)*(M*ε + ε*M) + mp.beta*I4*M;

    for i=1:3
        k = voigt_index[i][1]
        l = voigt_index[i][2]
        σ[i] = σmat[k,l];
    end

end

@views function material_routine!(ℂ,σ,ϵ,mp)

    #tangent_transiso!(ℂ,mp);
    stress_transiso!(σ,ϵ,mp);

end

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
        stress_transiso!(σ, ϵ, mp)

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


function NonlinearSolve(dh, cv, fv, ch, mp, u_d, ℂ,x, λf)

    _ndofs = ndofs(dh);
    u = zeros(_ndofs);
    Δu = zeros(_ndofs);
    ΔΔu = zeros(_ndofs);
    apply!(u_d, ch);
    # Create sparse matrix and residual vector
    K = create_sparsity_pattern(dh); #Create the sparsity pattern corresponding to the degree of freedom numbering in the DofHandler. Return a SparseMatrixCSC with stored values in the correct places.
    g = zeros(_ndofs);


    newton_itr = -1;
    NEWTON_TOL = 1e-10;
    normref = 1.0;

    
    while true; newton_itr += 1

        u .= u_d .+ Δu;      # updating displacements
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
        elseif normg> 1e15
            error("Residuum exploding $normg !!, aborting")
        end

        # Newton-Raphson: K * ΔΔu = g where g = f_ext - f_int
        # This gives the displacement correction
        ΔΔu = K \ g;
        apply_zero!(ΔΔu, ch)
        Δu .+= ΔΔu;  # Update displacement increment


    end

    return u

end;


@views function compute_results(cv, dh::DofHandler, u, mp_or_mf)

    ndofs = ndofs_per_cell(dh);
    cell_dofs = Vector{Int64}(undef,ndofs);
    nqp = getnquadpoints(cv);
    ue = Vector{Float64}(undef,ndofs);

    # Allocate storage for the fluxes to store
    σ = zeros(nqp,getncells(dh.grid),4);
    ϵ = zeros(nqp,getncells(dh.grid),4);

    σ_qp = Vector{Float64}(undef,4);        # Stress at a quad point
    ϵ_qp = Vector{Float64}(undef,4);
    Bmat = Array{Float64,2}(undef,3,ndofs_per_cell(dh));

    # Support either a global MaterialParams (mp_or_mf isa MaterialParams)
    # or a MaterialField (mp_or_mf isa MaterialField). When given a
    # MaterialField we compute an averaged MaterialParams for each element
    # and pass that into the stress routine so stresses are consistent with
    # the sampled field.
    is_field = isa(mp_or_mf, MaterialField)
    nloc_field = is_field ? size(mp_or_mf.μ_l, 2) : 1

    for (cell_num, cell) in enumerate(CellIterator(dh))
        cell_dofs .= celldofs(cell);
        ue .= u[cell_dofs]
        reinit!(cv, cell)

        # determine MaterialParams to use for this element
        if is_field
            mf = mp_or_mf
            μ_l_sum = 0.0; μ_t_sum = 0.0; α_sum = 0.0; β_sum = 0.0; λ_sum = 0.0; ang_sum = 0.0
            for ni in 1:nloc_field
                m = get_material(mf, cell_num, ni)
                μ_l_sum += Float64(m.μ_l)
                μ_t_sum += Float64(m.μ_t)
                α_sum   += Float64(m.α)
                β_sum   += Float64(m.β)
                λ_sum   += Float64(m.λ)
                ang_sum += Float64(m.angle)
            end
            nn = float(nloc_field)
            mp = MaterialParams(λ_sum/nn, μ_l_sum/nn, μ_t_sum/nn, α_sum/nn, β_sum/nn, ang_sum/nn)
        else
            mp = mp_or_mf
        end

        for qp in 1:nqp             #Iterate over all quadrature points (qp) (Integration points) for the current cell.
            stress_at_gp!(cv,qp,ue,σ_qp,ϵ_qp,Bmat,mp);
            σ[qp,cell_num,:] .= σ_qp
            ϵ[qp,cell_num,:] .= ϵ_qp
        end
    end

    return σ,ϵ
end

function stress_at_gp!(cv,qp,ue,σ_qp,ϵ_qp,Bmat,mp)      #calculate stress (σ) and strain (ϵ) at a specific quadrature point (qp) within a finite element cell

    fill!(Bmat,zero(eltype(Bmat)));
    fill!(σ_qp,zero(eltype(σ_qp)));
    fill!(ϵ_qp,zero(eltype(ϵ_qp)));
    
    for i=1:getnbasefunctions(cv)
        Bmat[1,2*i-1] = shape_gradient(cv, qp, i)[1];
        Bmat[2,2*i] = shape_gradient(cv, qp, i)[2];
        Bmat[3,2*i-1] = shape_gradient(cv, qp, i)[2];
        Bmat[3,2*i] = shape_gradient(cv, qp, i)[1];
    end
    ϵ_qp[1:3] .= Bmat*ue;

    stress_transiso!(σ_qp,ϵ_qp,mp)

    ϵ_qp[3] = ϵ_qp[3]/2;
    ϵ_qp[end] = ϵ_qp[3];

    σ_qp[end] = σ_qp[3];

    return nothing
end

function remove_files(save_path)
    # Be robust: ensure directory exists and attempt to remove matching files with warnings on failure
    if !isdir(save_path)
        @warn "remove_files: path does not exist: $save_path"
        return
    end
    for f in readdir(save_path, join=true)
        # Remove matching files in the top-level save_path
        if endswith(f, ".vtu") || endswith(f, ".pvd") || endswith(f, ".txt")
            try
                rm(f)
                @info "remove_files: removed $f"
            catch err
                @warn "remove_files: failed to remove $f: $err"
            end
        elseif isdir(f)
            # Remove per-run subdirectories (e.g., run_1, run_2) recursively
            name = basename(f)
            if startswith(name, "run_")
                try
                    rm(f; recursive=true, force=true)
                    @info "remove_files: removed directory $f"
                catch err
                    @warn "remove_files: failed to remove directory $f: $err"
                end
            else
                @info "remove_files: skipping directory $f (not matching run_*)"
            end
        end
    end
end

@views function export_vtk(u, dh, grid, cv, mp, ip, save_path, id; density=nothing)

    n_count = length(dh.grid.nodes)
    σ,ϵ = compute_results(cv, dh, u, mp);
    # If a topology density field is supplied, scale stresses by SIMP factor x^penal
    if density !== nothing
        # Expect density to be cell-wise (num_cells)
        nqp = size(σ, 1)
        ncell = size(σ, 2)
        if length(density) == ncell
            @inbounds for ci in 1:ncell
                sf = density[ci]^penal
                # scale all quadpoint stress components for this cell
                for qp in 1:nqp
                    σ[qp, ci, :] .*= sf
                end
            end
        else
            @warn "export_vtk: density length $(length(density)) != ncell $ncell; skipping stress scaling"
        end
    end
    projector = L2Projector(ip, grid);      # Projecting (Interpolating) the quadrature data back to the nodes
    σ_projected = zeros(n_count,4);         # Projected stress and strain initialised to zero
    ϵ_projected = zeros(n_count,4);

    for i=1:4
        σ_projected[:,i] = project(projector, σ[:,:,i], cv.qr; project_to_nodes=true)[:];
        ϵ_projected[:,i] = project(projector, ϵ[:,:,i], cv.qr; project_to_nodes=true)[:];
    end

    sigvec = [zero(Tensor{2, 2}) for _ in 1:n_count];               #Initialize arrays sigvec and strainvec to store stress and strain tensors, respectively, for each node.
    strainvec = [zero(Tensor{2, 2}) for _ in 1:n_count];
    for i=1:n_count
        sigvec[i] = Tensor{2,2}([σ_projected[i,1],σ_projected[i,4],σ_projected[i,3],σ_projected[i,2]])
        strainvec[i] = Tensor{2,2}([ϵ_projected[i,1],ϵ_projected[i,4],ϵ_projected[i,3],ϵ_projected[i,2]])
    end

    # Ensure output path is explicit and write a .vtu file per call. Using an explicit
    # filename avoids issues with WriteVTK collection indexing and makes it easy to
    # find the files on disk.
    fname = joinpath(save_path, string(id) * ".vtu")
    vtk_grid(fname, dh) do vtk
        vtk_point_data(vtk, dh, u)
        vtk_point_data(vtk, sigvec, "σ")
        vtk_point_data(vtk, strainvec, "ϵ")
        # optionally write cell-wise density (topology field)
        if density !== nothing
            try
                vtk_cell_data(vtk, density, "density")
            catch err
                @warn "export_vtk: failed to write density cell data: $err"
            end
        end
        vtk_save(vtk)
    end

    # Optionally write/update a PVD collection file. This is left out here to
    # avoid compatibility issues with WriteVTK collection indexing across
    # versions. If you need a .pvd, re-enable and ensure the collection API
    # in your WriteVTK version matches the usage.

end

function FE_Run!(ℂ, x_fe)
    #---
    Logging.with_logger(Logging.SimpleLogger(stderr, Logging.Warn)) do
    tick()

    nsteps=10;
    Δt = 1e-2;
    local_t = 0.0  # Use local time for load stepping
    u_local = zeros(ndofs(dh))  # Start from zero displacement each FE run
    
    for i=1:nsteps
        local_t += Δt;
        update!(ch, local_t);
        λf = i/nsteps
        u_local = NonlinearSolve(dh, cv, fv, ch, mp, u_local, ℂ, x_fe, λf);
        #exportresults(u_local, dh, grid, cv_post, mp, ip, save_path, i);      
    end
    
    global u = u_local  # Update global displacement
   
    tock()
    end
end



#TOPOPT HELPER functions
"""
Extract element nodal coordinates from meshgrid
"""
function get_element_coordinates(nodes::Matrix{Float64}, elements::Matrix{Int}, elem_id::Int)
    elem_nodes = elements[elem_id, :]
    elem_coords = zeros(4, 2)
    
    for i = 1:4
        elem_coords[i, :] = nodes[elem_nodes[i], :]
    end
    
    return elem_coords
end


# --- VTK Export Functions ---
"""
Export displacement (u) as cell data to Paraview

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


Export strain (epsilon) as cell data to Paraview

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


Export stress (sigma) as cell data to Paraview

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
end"""