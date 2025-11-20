function covariance_matrix_from_elemcoords(coords_elem::AbstractArray, σ::Float64, Lc::Float64;
                                          use_centroids::Bool=false,
                                          make_sparse::Bool=false,
                                          cutoff_mult::Float64=3.0,
                                          kernel::Symbol = :exponential,
                                          matern_nu::Float64 = 1.5,
                                          eltype_out=Float64)
    nelem, nloc = size(coords_elem)

    # collect points
    if use_centroids
        pts = Vector{typeof(coords_elem[1,1])}(undef, nelem)
        for ei in 1:nelem
            s = zero(coords_elem[1,1])
            for ni in 1:nloc
                s += coords_elem[ei, ni]
            end
            pts[ei] = s / nloc
        end
    else
        pts = Vector{typeof(coords_elem[1,1])}(undef, nelem * nloc)
        k = 1
        for ei in 1:nelem
            for ni in 1:nloc
                pts[k] = coords_elem[ei, ni]
                k += 1
            end
        end
    end

    n = length(pts)
    if n == 0
        return zeros(eltype_out,0,0), pts
    end

    cutoff = cutoff_mult * Lc

    kernel_val = function(r)

        if kernel == :exponential
            return (σ^2) * exp(-r / Lc)

        elseif kernel == :gaussian
            return (σ^2) * exp(-(r^2) / (2 * Lc^2))

        elseif kernel == :matern
            # simple Matern(ν) approximate using (ν=1.5 or 2.5 common)
            if isapprox(matern_nu, 1.5; atol=1e-8)
                s = sqrt(3.0) * r / Lc
                return (σ^2) * (1.0 + s) * exp(-s)
            elseif isapprox(matern_nu, 2.5; atol=1e-8)
                s = sqrt(5.0) * r / Lc
                return (σ^2) * (1.0 + s + (s^2)/3.0) * exp(-s)
            else
                # fallback to exponential
                return (σ^2) * exp(-r / Lc)
            end

        else
            return (σ^2) * exp(-r / Lc)

        end
    end

    if make_sparse
        I = Int[]; J = Int[]; V = eltype_out[]
        for i in 1:n
            pi = pts[i]
            for j in i:n
                r = norm(pi - pts[j])
                if r <= cutoff
                    push!(I, i); push!(J, j); push!(V, eltype_out(kernel_val(r)))
                    if i != j
                        push!(I, j); push!(J, i); push!(V, eltype_out(kernel_val(r)))
                    end
                end
            end
        end
        C = sparse(I, J, V, n, n)
    else
        C = zeros(eltype_out, n, n)
        for i in 1:n
            for j in i:n
                r = norm(pts[i] - pts[j])
                val = kernel_val(r)
                C[i,j] = eltype_out(val)
                C[j,i] = C[i,j]
            end
        end
    end

    return C, pts
end

function KL_realization(material_params::MaterialParams, coords_elem::AbstractArray;
                        σs=Dict{Symbol,Float64}(), Lc=0.1, N_modes=5,
                        use_centroids=false, make_sparse=true, eltype_out=Float32,
                        kernel::Symbol = :gaussian, matern_nu::Float64=1.5,
                        mode::Symbol = :additive, seed::Union{Nothing,Integer}=nothing)
    """
    Generate Karhunen–Loève (KL) realizations for selectable scalar material fields.

    Arguments
    - material_params: MaterialParams containing mean property values.
    - coords_elem: element/node coordinates array with shape (n_elem, n_loc).

    Keyword arguments
    - σs: Dict mapping property symbols (e.g. :μ_l) to desired std-dev for the covariance.
    - Lc: correlation length.
    - N_modes: requested number of KL modes (truncated if larger than dof count).
    - use_centroids: if true, generate one sample per element (centroid); otherwise per node.
    - make_sparse: ask covariance builder to return a sparse covariance (helps large meshes).
    - eltype_out: element type for covariance entries (Float32/64).
    - kernel, matern_nu: covariance kernel controls forwarded to covariance_matrix_from_elemcoords.
    - mode: :additive (field = mean + KL) or :lognormal (field = mean * exp(KL)).
    - seed: Random seed for reproducibility. If nothing, uses current RNG state.

    Returns a Dict{Symbol,Any} where each key is a material property symbol and values are
    either vectors (if use_centroids) or arrays sized (n_elem, n_loc) matching coords_elem.
    """
    
    # Set seed for reproducibility if provided
    if !isnothing(seed)
        Random.seed!(seed)
    end

    n_elem, n_loc = size(coords_elem)
    result_fields = Dict{Symbol,Any}()

    # Properties to sample (symbol => mean_value)
    properties = (
        :μ_l => material_params.μ_l,
        :μ_t => material_params.μ_t,
        :α   => material_params.alpha,
        :β   => material_params.beta,
    )

    for (prop_sym, mean_value) in properties
        # choose sigma: either provided or a reasonable default relative to the mean
        default_sigma = 0.25 * abs(mean_value)
        sigma = get(σs, prop_sym, default_sigma)

        cov_matrix, points = covariance_matrix_from_elemcoords(coords_elem, sigma, Lc;
                                                              use_centroids=use_centroids,
                                                              make_sparse=make_sparse,
                                                              cutoff_mult=3.0,
                                                              kernel=kernel,
                                                              matern_nu=matern_nu,
                                                              eltype_out=eltype_out)

        n_dofs = size(cov_matrix, 1)
        if n_dofs == 0
            # degenerate case: no points -> constant field
            result_fields[prop_sym] = use_centroids ? fill(mean_value, n_elem) : fill(mean_value, n_elem, n_loc)
            continue
        end

        # Determine how many eigenpairs to compute. For ARPACK, nev must be < n_dofs.
        n_requested = min(N_modes, n_dofs)
        arpack_nev = min(n_requested, max(1, n_dofs - 1))

        eigenvals = nothing
        eigenvecs = nothing

        # Prefer dense symmetric eigen decomposition for small dense problems (fast & robust)
        if n_dofs <= 2000 && !issparse(cov_matrix)
            # ensure a standard dense symmetric matrix for eigen
            denseC = Matrix{Float64}(cov_matrix)
            ev = eigen(Symmetric(denseC))
            # eigen returns ascending order; take largest n_requested
            idx_desc = sortperm(ev.values, rev=true)[1:n_requested]
            eigenvals = ev.values[idx_desc]
            eigenvecs = ev.vectors[:, idx_desc]
        else
            # Try ARPACK for large / sparse problems. If it fails, fall back to dense eigen.
            try
                # load Arpack lazily; using inside try avoids hard dependency at module load
                @eval begin
                    using Arpack
                end
                # Arpack returns nev eigenpairs; request arpack_nev (must be < n_dofs)
                arpack_vals, arpack_vecs = Arpack.eigs(cov_matrix; nev=arpack_nev, which=:LM)
                eigenvals = real(arpack_vals)
                eigenvecs = real(arpack_vecs)
                # If ARPACK returned fewer modes than requested, we may truncate later
            catch err
                @warn "ARPACK eigs failed, falling back to dense eigen: $err"
                denseC = Matrix{Float64}(cov_matrix)
                ev = eigen(Symmetric(denseC))
                idx_desc = sortperm(ev.values, rev=true)[1:n_requested]
                eigenvals = ev.values[idx_desc]
                eigenvecs = ev.vectors[:, idx_desc]
            end
        end

        # Truncate to the final requested number of modes (ARPACK may have returned less)
        n_available = length(eigenvals)
        n_modes_final = min(n_requested, n_available)
        eigenvals = eigenvals[1:n_modes_final]
        eigenvecs = eigenvecs[:, 1:n_modes_final]

        # Numerical safety: clamp tiny negative eigenvalues to zero
        eigenvals = max.(eigenvals, zero(real(eigenvals[1])))

        # sample standard normal coefficients and build the KL field
        coeffs = randn(n_modes_final)
        mode_amplitudes = sqrt.(eigenvals) .* coeffs
        gaussian_field = eigenvecs * mode_amplitudes

        if mode == :additive
            sampled_values = mean_value .+ gaussian_field
        else
            sampled_values = mean_value .* exp.(gaussian_field)
        end

        # Map back to element/node layout
        if use_centroids
            result_fields[prop_sym] = convert(Array{eltype_out,1}, sampled_values)
        else
            field_array = Array{Float64}(undef, n_elem, n_loc)
            k = 1
            for ei in 1:n_elem, ni in 1:n_loc
                field_array[ei, ni] = float(sampled_values[k])
                k += 1
            end
            result_fields[prop_sym] = field_array
        end
    end

    # Copy through parameters left unchanged (λ and angle)
    result_fields[:λ] = use_centroids ? fill(material_params.λ, n_elem) : fill(material_params.λ, n_elem, n_loc)
    result_fields[:angle] = use_centroids ? fill(material_params.angle, n_elem) : fill(material_params.angle, n_elem, n_loc)

    return result_fields
end


