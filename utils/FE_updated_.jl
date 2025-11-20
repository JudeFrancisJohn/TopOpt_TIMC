struct MaterialParams
    λ::Float64
    μ_l::Float64
    μ_t::Float64
    alpha::Float64
    beta::Float64
    angle::Float64
end

@with_kw struct NeumannpointBoundaryinfo
    ΓNlist::Vector{Set};
    tractionlist::Vector{Vector{Float64}};
end

@views function trans_iso_tangent!(ℂ,mp)

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
@views function tangent_transverse_isotropy!(λ::T, μ_l::T, μ_t::T, alpha::T, beta::T, angle::T) where {T}
    voigt_index = [[1,1],[2,2],[1,2]]
    δ = Matrix(1.0I, 2, 2)
    M = similar(δ)
    rot_angle = deg2rad(angle)
    a = [cos(rot_angle),sin(rot_angle)]
    for i=1:2
        for j=1:2
            M[i,j] = a[i]*a[j];
        end
    end

    ℂ = zeros(T, 2, 2, 2, 2)
    II = similar(ℂ);
    IIa = similar(ℂ);
    for i=1:2
        for j=1:2
            for k=1:2
                for l=1:2
                    II[i,j,k,l] = (1/2)*(δ[i,k] * δ[j,l] + δ[i,l] * δ[j,k]);
                    IIa[i,j,k,l] = (1/2)*(M[i,k]*δ[j,l] + M[i,l]*δ[j,k] + M[j,l]*δ[i,k] + M[j,k]*δ[i,l]);
                    ℂ[i,j,k,l] = (λ * δ[i,j] * δ[k,l] + 2.0 * μ_t
             * II[i,j,k,l] + alpha*(δ[i,j]*M[k,l] + M[i,j]*δ[k,l])
                                    + 2.0*(μ_l-μ_t
                            )*IIa[i,j,k,l] + beta*M[i,j]*M[k,l]);
                end
            end
        end
    end
     # Fill only upper triangle of Voigt matrix
    C = zeros(T, 3, 3)
    for i in 1:3
        k, l = voigt_index[i]
        for j in i:3   # upper triangle only
            p, q = voigt_index[j]
            C[i, j] = ℂ[k, l, p, q]
        end
    end
    for i in 1:3
        for j in i+1:3
            C[j, i] = C[i, j]
        end
    end
    return C
end

@views function stress_transverseiso!(σ,ϵ,mp)
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



@views function compute_results(cv, dh::DofHandler, u, mp)

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

    for (cell_num, cell) in enumerate(CellIterator(dh))
        cell_dofs .= celldofs(cell);
        ue .= u[cell_dofs]
        reinit!(cv, cell)
        for qp in 1:nqp             #Iterate over all quadrature points (qp) (Integration points) for the current cell.
            sigmaatqp!(cv,qp,ue,σ_qp,ϵ_qp,Bmat,mp);
            σ[qp,cell_num,:] .= σ_qp
            ϵ[qp,cell_num,:] .= ϵ_qp
        end
    end

    return σ,ϵ
end

function sigmaatqp!(cv,qp,ue,σ_qp,ϵ_qp,Bmat,mp)      #calculate stress (σ) and strain (ϵ) at a specific quadrature point (qp) within a finite element cell

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

    trans_iso_stress!(σ_qp,ϵ_qp,mp)

    ϵ_qp[3] = ϵ_qp[3]/2;
    ϵ_qp[end] = ϵ_qp[3];

    σ_qp[end] = σ_qp[3];

    return nothing
end

function clean_savepath(save_path)
    foreach(rm, filter(endswith(".vtu"), readdir(save_path,join=true)));
    foreach(rm, filter(endswith(".pvd"), readdir(save_path,join=true)));
    foreach(rm, filter(endswith(".txt"), readdir(save_path,join=true)));
end

@views function exportresults(u, dh, grid, cv, mp, ip, save_path, id)

    n_count = length(dh.grid.nodes)
    σ,ϵ = compute_results(cv, dh, u, mp);
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

    pvd = paraview_collection(save_path*"/TopOpt.pvd",append=true);
    vtk_grid(save_path*"/$id", dh) do vtk
        vtk_point_data(vtk, dh, u)
        vtk_point_data(vtk, sigvec, "σ")
        vtk_point_data(vtk, strainvec, "ϵ")
        vtk_save(vtk)
        pvd[id] = vtk
    end
    vtk_save(pvd);

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