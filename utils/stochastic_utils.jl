function covariance_matrix_from_elemcoords(coords_elem::AbstractArray, σ::Float64, Lc::Float64;
    use_centroids::Bool=false,
    make_sparse::Bool=false,
    cutoff_mult::Float64=3.0,
    kernel::Symbol=:exponential,
    matern_nu::Float64=1.5,
    eltype_out=Float64)
    nelem, nloc = size(coords_elem)

    # collect points
    if use_centroids
        pts = Vector{typeof(coords_elem[1, 1])}(undef, nelem)
        for ei in 1:nelem
            s = zero(coords_elem[1, 1])
            for ni in 1:nloc
                s += coords_elem[ei, ni]
            end
            pts[ei] = s / nloc
        end
    else
        pts = Vector{typeof(coords_elem[1, 1])}(undef, nelem * nloc)
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
        return zeros(eltype_out, 0, 0), pts
    end

    cutoff = cutoff_mult * Lc

    kernel_val = function (r)

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
                return (σ^2) * (1.0 + s + (s^2) / 3.0) * exp(-s)
            else
                # fallback to exponential
                return (σ^2) * exp(-r / Lc)
            end

        else
            return (σ^2) * exp(-r / Lc)

        end
    end

    if make_sparse
        I = Int[]
        J = Int[]
        V = eltype_out[]
        for i in 1:n
            pi = pts[i]
            for j in i:n
                r = norm(pi - pts[j])
                if r <= cutoff
                    push!(I, i)
                    push!(J, j)
                    push!(V, eltype_out(kernel_val(r)))
                    if i != j
                        push!(I, j)
                        push!(J, i)
                        push!(V, eltype_out(kernel_val(r)))
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
                C[i, j] = eltype_out(val)
                C[j, i] = C[i, j]
            end
        end
    end

    return C, pts
end

"""
Struct holding precomputed KL expansion eigenmodes for a single material property.
This separates the expensive eigenvalue problem (solved once based on mean/covariance)
from coefficient sampling (done per realization).
"""
struct KL_Eigenmodes
    property_symbol::Symbol
    mean_value::Float64
    eigenvalues::Vector{Float64}
    eigenvectors::Matrix{Float64}
    n_elem::Int
    n_loc::Int
    use_centroids::Bool
    mode::Symbol  # :additive or :lognormal
end

"""
    compute_KL_eigenmodes(material_params, coords_elem, prop_sym, sigma; kwargs...)

Solve the KL eigenvalue problem ONCE for a given material property.
This is the deterministic part that depends only on:
- Mean value of the property
- Covariance structure (sigma, Lc, kernel)
- Geometry (coords_elem)

Returns a KL_Eigenmodes struct containing the eigenmodes and eigenvalues.
"""
function compute_KL_eigenmodes(material_params::MaterialParams, 
                               coords_elem::AbstractArray,
                               prop_sym::Symbol,
                               sigma::Float64;
                               Lc=0.1, 
                               N_modes=5,
                               use_centroids=false, 
                               make_sparse=true, 
                               eltype_out=Float32,
                               kernel::Symbol=:gaussian, 
                               matern_nu::Float64=1.5,
                               mode::Symbol=:additive)
    
    # Get mean value for this property
    mean_value = if prop_sym == :μ_l
        material_params.μ_l
    elseif prop_sym == :μ_t
        material_params.μ_t
    elseif prop_sym == :α
        material_params.alpha
    elseif prop_sym == :β
        material_params.beta
    else
        error("Unknown property symbol: $prop_sym")
    end

    n_elem, n_loc = size(coords_elem)

    # Build covariance matrix (deterministic based on mean/geometry/sigma)
    cov_matrix, points = covariance_matrix_from_elemcoords(coords_elem, sigma, Lc;
        use_centroids=use_centroids,
        make_sparse=make_sparse,
        cutoff_mult=3.0,
        kernel=kernel,
        matern_nu=matern_nu,
        eltype_out=eltype_out)

    n_dofs = size(cov_matrix, 1)
    if n_dofs == 0
        # Degenerate case: return empty eigenmodes
        return KL_Eigenmodes(prop_sym, mean_value, Float64[], zeros(Float64, 0, 0), 
                           n_elem, n_loc, use_centroids, mode)
    end

    # Determine how many eigenpairs to compute
    n_requested = min(N_modes, n_dofs)
    arpack_nev = min(n_requested, max(1, n_dofs - 1))

    eigenvals = nothing
    eigenvecs = nothing

    # Solve eigenvalue problem (DETERMINISTIC - only depends on covariance structure)
    if n_dofs <= 2000 && !issparse(cov_matrix)
        denseC = Matrix{Float64}(cov_matrix)
        ev = eigen(Symmetric(denseC))
        idx_desc = sortperm(ev.values, rev=true)[1:n_requested]
        eigenvals = ev.values[idx_desc]
        eigenvecs = ev.vectors[:, idx_desc]
    else
        try
            @eval begin
                using Arpack
            end
            arpack_vals, arpack_vecs = Arpack.eigs(cov_matrix; nev=arpack_nev, which=:LM)
            eigenvals = real(arpack_vals)
            eigenvecs = real(arpack_vecs)
        catch err
            @warn "ARPACK eigs failed, falling back to dense eigen: $err"
            denseC = Matrix{Float64}(cov_matrix)
            ev = eigen(Symmetric(denseC))
            idx_desc = sortperm(ev.values, rev=true)[1:n_requested]
            eigenvals = ev.values[idx_desc]
            eigenvecs = ev.vectors[:, idx_desc]
        end
    end

    # Truncate to final number of modes
    n_available = length(eigenvals)
    n_modes_final = min(n_requested, n_available)
    eigenvals = eigenvals[1:n_modes_final]
    eigenvecs = eigenvecs[:, 1:n_modes_final]

    # Numerical safety: clamp tiny negative eigenvalues to zero
    eigenvals = max.(eigenvals, zero(real(eigenvals[1])))

    return KL_Eigenmodes(prop_sym, mean_value, eigenvals, eigenvecs, 
                        n_elem, n_loc, use_centroids, mode)
end

"""
    sample_KL_field(kl_modes, coeffs; eltype_out)

Generate a single realization from precomputed KL eigenmodes using provided coefficients.
This is the RANDOM part - only the coefficients vary between realizations.

Arguments:
- kl_modes: KL_Eigenmodes struct from compute_KL_eigenmodes
- coeffs: Vector of standard normal coefficients (length = number of modes)
- eltype_out: output element type (Float32/Float64)

Returns the sampled field as either a vector (if use_centroids) or array (n_elem, n_loc).
"""
function sample_KL_field(kl_modes::KL_Eigenmodes, 
                        coeffs::Vector{Float64};
                        eltype_out=Float32)
    
    n_modes = length(kl_modes.eigenvalues)
    
    # Handle coefficient dimension mismatch
    if length(coeffs) < n_modes
        # Pad with zeros if too few coefficients provided
        coeffs_padded = zeros(n_modes)
        coeffs_padded[1:length(coeffs)] = coeffs
        coeffs = coeffs_padded
    elseif length(coeffs) > n_modes
        # Truncate if too many coefficients provided
        coeffs = coeffs[1:n_modes]
    end

    # KL expansion: field = mean + Σ sqrt(λᵢ) * ξᵢ * φᵢ
    mode_amplitudes = sqrt.(kl_modes.eigenvalues) .* coeffs
    gaussian_field = kl_modes.eigenvectors * mode_amplitudes

    # Apply transformation (additive or lognormal)
    if kl_modes.mode == :additive
        sampled_values = kl_modes.mean_value .+ gaussian_field
    else  # :lognormal
        sampled_values = kl_modes.mean_value .* exp.(gaussian_field)
    end

    # Map back to element/node layout
    if kl_modes.use_centroids
        return convert(Array{eltype_out,1}, sampled_values)
    else
        field_array = Array{Float64}(undef, kl_modes.n_elem, kl_modes.n_loc)
        k = 1
        for ei in 1:kl_modes.n_elem, ni in 1:kl_modes.n_loc
            field_array[ei, ni] = float(sampled_values[k])
            k += 1
        end
        return field_array
    end
end

"""
    KL_realization(material_params, coords_elem; kwargs...)

LEGACY WRAPPER: Generate KL realizations by computing eigenmodes and sampling in one call.
This maintains backward compatibility but is less efficient for multiple realizations.

For MCMC or multiple runs, prefer:
1. Call compute_KL_eigenmodes() ONCE for each property
2. Call sample_KL_field() for each realization with different coefficients

Arguments:
- material_params: MaterialParams containing mean property values.
- coords_elem: element/node coordinates array with shape (n_elem, n_loc).

Keyword arguments:
- σs: Dict mapping property symbols (e.g. :μ_l) to desired std-dev for the covariance.
- Lc: correlation length.
- N_modes: requested number of KL modes (truncated if larger than dof count).
- use_centroids: if true, generate one sample per element (centroid); otherwise per node.
- make_sparse: ask covariance builder to return a sparse covariance (helps large meshes).
- eltype_out: element type for covariance entries (Float32/64).
- kernel, matern_nu: covariance kernel controls forwarded to covariance_matrix_from_elemcoords.
- mode: :additive (field = mean + KL) or :lognormal (field = mean * exp(KL)).
- seed: Random seed for reproducibility. If nothing, uses current RNG state.
- provided_coeffs: Optional Dict mapping property symbols to coefficient vectors. 
                   If provided, these coefficients are used instead of random sampling.
                   Useful for MCMC or reconstructing specific realizations.

Returns a Dict{Symbol,Any} where each key is a material property symbol and values are
either vectors (if use_centroids) or arrays sized (n_elem, n_loc) matching coords_elem.
"""
function KL_realization(material_params::MaterialParams, coords_elem::AbstractArray;
    σs=Dict{Symbol,Float64}(), Lc=0.1, N_modes=5,
    use_centroids=false, make_sparse=true, eltype_out=Float32,
    kernel::Symbol=:gaussian, matern_nu::Float64=1.5,
    mode::Symbol=:additive, seed::Union{Nothing,Integer}=nothing,
    provided_coeffs::Union{Nothing,Dict{Symbol,Vector{Float64}}}=nothing)

    # Set seed for reproducibility if provided
    if !isnothing(seed)
        Random.seed!(seed)
    end

    n_elem, n_loc = size(coords_elem)
    result_fields = Dict{Symbol,Any}()

    # Properties to sample (symbol => default sigma multiplier)
    properties = (:μ_l, :μ_t, :α, :β)

    for prop_sym in properties
        # Choose sigma: either provided or a reasonable default
        mean_value = if prop_sym == :μ_l
            material_params.μ_l
        elseif prop_sym == :μ_t
            material_params.μ_t
        elseif prop_sym == :α
            material_params.alpha
        elseif prop_sym == :β
            material_params.beta
        end
        
        default_sigma = 0.25 * abs(mean_value)
        sigma = get(σs, prop_sym, default_sigma)

        # Compute eigenmodes (deterministic)
        kl_modes = compute_KL_eigenmodes(material_params, coords_elem, prop_sym, sigma;
            Lc=Lc, N_modes=N_modes, use_centroids=use_centroids,
            make_sparse=make_sparse, eltype_out=eltype_out,
            kernel=kernel, matern_nu=matern_nu, mode=mode)

        # Handle degenerate case
        if length(kl_modes.eigenvalues) == 0
            result_fields[prop_sym] = use_centroids ? fill(mean_value, n_elem) : fill(mean_value, n_elem, n_loc)
            continue
        end

        # Sample coefficients (random or provided)
        n_modes_final = length(kl_modes.eigenvalues)
        if !isnothing(provided_coeffs) && haskey(provided_coeffs, prop_sym)
            coeffs = provided_coeffs[prop_sym]
        else
            coeffs = randn(n_modes_final)
        end

        # Generate field from eigenmodes and coefficients
        result_fields[prop_sym] = sample_KL_field(kl_modes, coeffs; eltype_out=eltype_out)
    end

    # Copy through parameters left unchanged (λ and angle)
    result_fields[:λ] = use_centroids ? fill(material_params.λ, n_elem) : fill(material_params.λ, n_elem, n_loc)
    result_fields[:angle] = use_centroids ? fill(material_params.angle, n_elem) : fill(material_params.angle, n_elem, n_loc)

    return result_fields
end


