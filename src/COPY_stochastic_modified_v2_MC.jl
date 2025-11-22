using LinearAlgebra, Printf, SparseArrays, StaticArrays
using Plots, Dates, Statistics, DelimitedFiles
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
include("../input/params_LOGS.jl")
include("geom_BC.jl")
mp = MaterialParams(λ, μ_l, μ_t, alpha, beta, angle)

run_mode = "stochastic"
dt = Dates.format(Dates.now(), "yyyymmdd_HH")
run_name = "$(run_mode)_$(dt)"   # change this string to a custom run name if desired

const WRITE_OUTPUT_FILES = false


save_root = joinpath(@__DIR__, "..", "output")
save_path = joinpath(save_root, run_name)
if WRITE_OUTPUT_FILES
    mkpath(save_path)
    remove_files(save_path)
    export_vtk(u_d, dh, grid, cv_post, mp, ip, save_path, 0)
end

cells = getcells(grid)
elements = [cell.nodes for cell in cells]
nodes = getnodes(grid)
nelem = size(elements)
nnodes = size(nodes)

#Geometry and element/node counts
dim = 2
nnodes_per_cell = length(cells[1].nodes)   # 4 for quadrilateral (nodes per element)
nelem = length(cells)
C = zeros(3, 3)
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
ϵ = Vector{Float64}(undef, 3)
σ = similar(ϵ)
ℂ = Array{Float64}(undef, 3, 3)
KE_store = Vector{Matrix{Float64}}(undef, nelem)

num_cells = length(CellIterator(dh))
if num_cells == (nelx * nely)
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

const DENSITY_BIN_EDGES = (0.0, LOG_BIN_LOW, LOG_BIN_MID, LOG_BIN_HIGH)
const DensityLogEntry = NamedTuple{(:iteration, :bin_counts, :bin_proportions, :poor_design),Tuple{Int,NTuple{3,Int},NTuple{3,Float64},Bool}}
const ShearStat = NamedTuple{(:mean, :std, :cov),Tuple{Float64,Float64,Float64}}
const ShearStats = NamedTuple{(:μ_l, :μ_t),Tuple{ShearStat,ShearStat}}
const RunLogEntry = NamedTuple{(:run_index, :status, :error, :compliance, :density_history, :final_density_log, :final_volume, :shear_stats, :resample_attempts, :material_seed, :output_dir, :vtu_file, :log_file),Tuple{Int,Symbol,Union{Nothing,String},Float64,Vector{DensityLogEntry},Union{DensityLogEntry,Nothing},Float64,ShearStats,Int,Union{Nothing,Int},String,Union{Nothing,String},Union{Nothing,String}}}
const MATERIAL_SEED_BASE = 1000

function write_run_log(run_dir::AbstractString; status::Symbol, shear_stats::ShearStats, resample_attempts::Int, material_seed, density_log, compliance::Float64, final_volume::Float64, final_change, guard_threshold::Float64, max_resamples::Int, enforce_guard::Bool, abort_on_guard::Bool, vtu_file, error::Union{Nothing,String})
    mkpath(run_dir)
    log_path = joinpath(run_dir, "run_log.txt")
    seed_info = isnothing(material_seed) ? "N/A" : string(material_seed)
    change_info = isnothing(final_change) ? "N/A" : string(final_change)
    open(log_path, "w") do io
        println(io, "status=$(status)")
        println(io, "resample_attempts=$(resample_attempts)")
        println(io, "material_seed=$(seed_info)")
        println(io, "guard_threshold=$(guard_threshold)")
        println(io, "guard_enforced=$(enforce_guard)")
        println(io, "abort_on_guard=$(abort_on_guard)")
        println(io, "max_resamples=$(max_resamples)")
        println(io, "shear_stats_mu_l_mean=$(shear_stats.μ_l.mean)")
        println(io, "shear_stats_mu_l_std=$(shear_stats.μ_l.std)")
        println(io, "shear_stats_mu_l_cov=$(shear_stats.μ_l.cov)")
        println(io, "shear_stats_mu_t_mean=$(shear_stats.μ_t.mean)")
        println(io, "shear_stats_mu_t_std=$(shear_stats.μ_t.std)")
        println(io, "shear_stats_mu_t_cov=$(shear_stats.μ_t.cov)")
        println(io, "final_compliance=$(compliance)")
        println(io, "final_volume=$(final_volume)")
        println(io, "final_change=$(change_info)")
        if density_log !== nothing
            println(io, "density_bin_counts=$(density_log.bin_counts)")
            println(io, "density_bin_proportions=$(density_log.bin_proportions)")
            println(io, "density_poor_flag=$(density_log.poor_design)")
            println(io, "final_iteration=$(density_log.iteration)")
        end
        println(io, "vtu_file=$(isnothing(vtu_file) ? "N/A" : vtu_file)")
        if error !== nothing
            println(io, "error=$(error)")
        end
    end
    return log_path
end

function compute_density_metrics(x::AbstractVector{<:Real})
    total = length(x)
    total == 0 && error("Density vector is empty")
    c1 = 0
    c2 = 0
    c3 = 0
    for val in x
        v = clamp(Float64(val), DENSITY_BIN_EDGES[1], DENSITY_BIN_EDGES[end])
        if v < DENSITY_BIN_EDGES[2]
            c1 += 1
        elseif v < DENSITY_BIN_EDGES[3]
            c2 += 1
        else
            c3 += 1
        end
    end
    counts = (c1, c2, c3)
    inv_total = 1.0 / total
    proportions = (c1 * inv_total, c2 * inv_total, c3 * inv_total)
    return counts, proportions
end

function log_density_metrics!(storage::Vector{DensityLogEntry}, iteration::Int, x::AbstractVector{<:Real})
    counts, proportions = compute_density_metrics(x)
    poor_flag = proportions[2] > LOG_BAD_BIN_THRESHOLD
    push!(storage, (iteration=iteration,
        bin_counts=counts,
        bin_proportions=proportions,
        poor_design=poor_flag))
    return poor_flag
end

flatten_field_values(data) = Float64.(vec(data))

function compute_shear_stat(values::AbstractVector{<:Real})
    m = mean(values)
    s = std(values)
    denom = max(abs(m), eps())
    cov = s / denom
    return (mean=m, std=s, cov=cov)::ShearStat
end

function shear_stats_from_fields(fields::Dict{Symbol,Any})
    haskey(fields, :μ_l) || error("μ_l field missing from stochastic realization")
    haskey(fields, :μ_t) || error("μ_t field missing from stochastic realization")
    μ_l_vals = flatten_field_values(fields[:μ_l])
    μ_t_vals = flatten_field_values(fields[:μ_t])
    return (μ_l=compute_shear_stat(μ_l_vals), μ_t=compute_shear_stat(μ_t_vals))::ShearStats
end

function shear_variation_ok(stats::ShearStats)
    threshold = LOG_SHEAR_COV_THRESHOLD
    return stats.μ_l.cov <= threshold && stats.μ_t.cov <= threshold
end

function sample_material_fields(run_index)
    max_attempts = max(LOG_SHEAR_MAX_RESAMPLES, 1)
    fields = Dict{Symbol,Any}()
    stats = (μ_l=(mean=0.0, std=0.0, cov=Inf), μ_t=(mean=0.0, std=0.0, cov=Inf))::ShearStats
    seed_used = 0
    for attempt in 1:max_attempts
        seed = MATERIAL_SEED_BASE + (run_index - 1) * max_attempts + attempt - 1
        candidate_fields = KL_realization(mp, coords_elem;
            σs=Dict(:μ_l => 0.5,
                :μ_t => 0.5,
                :α => 0.5,
                :β => 0.5),
            Lc=0.01, N_modes=80, use_centroids=false,
            make_sparse=true, kernel=:exponential, mode=:lognormal,
            seed=seed)
        candidate_stats = shear_stats_from_fields(candidate_fields)
        seed_used = seed
        fields = candidate_fields
        stats = candidate_stats
        if !LOG_ENFORCE_SHEAR_VARIATION || shear_variation_ok(candidate_stats)
            return true, fields, stats, attempt, seed_used
        end
    end
    println("[WARN] Shear variability constraint not met after $(max_attempts) attempts; last sample seed=$(seed_used).")
    if LOG_ABORT_ON_SHEAR_FAILURE
        return false, fields, stats, max_attempts, seed_used
    end
    return true, fields, stats, max_attempts, seed_used
end

function topopt_run(run_i, shear_stats::ShearStats, resample_attempts::Int, material_seed::Union{Nothing,Int})
    x = ones(num_cells)
    change = 1.0
    loop = 0
    vol_history = Float64[]
    last_compliance = 0.0
    status = :success
    err_msg::Union{Nothing,String} = nothing
    final_vtu::Union{Nothing,String} = nothing

    run_dir = WRITE_OUTPUT_FILES ? joinpath(save_path, "run_$(run_i)") : ""
    final_dir = WRITE_OUTPUT_FILES ? joinpath(run_dir, "final") : ""
    if WRITE_OUTPUT_FILES
        mkpath(final_dir)
    end

    density_history = Vector{DensityLogEntry}()
    log_density_metrics!(density_history, loop, x)

    while change > 0.01 && loop < 1000
        current_iter = loop + 1
        xold = copy(x)
        c = 0.0
        dc = zeros(num_cells)
        iteration_ok = true

        try
            FE_Run!(ℂ, x)

            for (cell_index, cell) in enumerate(CellIterator(dh))
                local global_dofs = celldofs(cell)
                local ue = u[global_dofs]
                local ke = KE_store[cell_index]

                x_cell = x[cell_index]
                c += x_cell^penal * (ue' * ke * ue)
                dc[cell_index] = -penal * x_cell^(penal - 1) * (ue' * ke * ue)
            end

            dc = check(nelx, nely, rmin, x, dc)
            dc = max.(dc, -abs.(dc) * 1e-12)

            x = OC(x, volfrac, dc)
        catch err
            status = :failed
            err_msg = sprint(showerror, err)
            iteration_ok = false
            x = xold
        end

        if !iteration_ok
            log_density_metrics!(density_history, current_iter, x)
            break
        end

        actual_volfrac = sum(x) / num_cells
        change = maximum(abs.(x .- xold))
        push!(vol_history, actual_volfrac)

        last_compliance = c
        loop = current_iter
        log_density_metrics!(density_history, loop, x)
    end

    @assert !isempty(density_history)
    final_density_log = density_history[end]
    final_volume = sum(x) / num_cells
    final_iter = final_density_log.iteration

    final_name = ""
    final_vtu_path = ""
    if WRITE_OUTPUT_FILES
        final_name = @sprintf("run_%d_%s_iter_%03d", run_i, string(status), final_iter)
        final_vtu_path = joinpath(final_dir, final_name * ".vtu")
        export_vtk(u, dh, grid, cv_post, mp, ip, final_dir, final_name; density=x)
    end
    final_vtu = WRITE_OUTPUT_FILES ? final_vtu_path : nothing

    println("\n" * "="^80)
    println("║" * " "^23 * "TOPOLOGY OPTIMIZATION COMPLETE" * " "^23 * "║")
    println("="^80)
    println("Final Results:")
    println("  • Status: $(status)")
    println("  • Completed iterations: $loop")
    println("  • Final objective: $(round(last_compliance, digits=4))")
    #println("  • Final volume fraction: $(round(final_volume, digits=4))")
    #println("  • Final change: $(round(change, digits=6))")
    println("  • μ_l stats: mean=$(round(shear_stats.μ_l.mean, digits=4)), std=$(round(shear_stats.μ_l.std, digits=4)), cov=$(round(shear_stats.μ_l.cov, digits=4))")
    println("  • μ_t stats: mean=$(round(shear_stats.μ_t.mean, digits=4)), std=$(round(shear_stats.μ_t.std, digits=4)), cov=$(round(shear_stats.μ_t.cov, digits=4))")
    seed_info = isnothing(material_seed) ? "N/A" : string(material_seed)
    println("  • Material seed used: $(seed_info) (resample attempts: $(resample_attempts))")
    counts = final_density_log.bin_counts
    proportions = map(p -> round(p, digits=3), final_density_log.bin_proportions)
    println("  • Density bin counts: $(counts)")
    println("  • Density bin proportions: $(proportions)")
    println("  • Poor design flag: $(final_density_log.poor_design)")
    output_info = WRITE_OUTPUT_FILES ? final_vtu_path : "N/A (files disabled)"
    println("  • Output file: $(output_info)")
    if err_msg !== nothing
        println("  • Error: $(err_msg)")
    end
    println("="^80)

    log_path = WRITE_OUTPUT_FILES ? write_run_log(run_dir;
        status=status,
        shear_stats=shear_stats,
        resample_attempts=resample_attempts,
        material_seed=material_seed,
        density_log=final_density_log,
        compliance=last_compliance,
        final_volume=final_volume,
        final_change=change,
        guard_threshold=LOG_SHEAR_COV_THRESHOLD,
        max_resamples=LOG_SHEAR_MAX_RESAMPLES,
        enforce_guard=LOG_ENFORCE_SHEAR_VARIATION,
        abort_on_guard=LOG_ABORT_ON_SHEAR_FAILURE,
        vtu_file=final_vtu,
        error=err_msg) : nothing

    return (run_index=run_i,
        status=status,
        error=err_msg,
        compliance=last_compliance,
        density_history=density_history,
        final_density_log=final_density_log,
        final_volume=final_volume,
        shear_stats=shear_stats,
        resample_attempts=resample_attempts,
        material_seed=material_seed,
    output_dir=final_dir,
        vtu_file=final_vtu,
        log_file=log_path)
end


function multiple_runs(nruns=3)
    run_logs = Vector{RunLogEntry}()

    for i in 1:nruns
        success, fields, shear_stats, attempts, seed_used = sample_material_fields(i)
        if success
            if LOG_ENFORCE_SHEAR_VARIATION
                println("RUN  - $(i) | shear cov μ_l=$(round(shear_stats.μ_l.cov, digits=4)), μ_t=$(round(shear_stats.μ_t.cov, digits=4)); attempts=$(attempts)")
            else
                println("RUN  - $(i)")
            end

            mf = build_material_field(fields; use_centroids=false, eltype_out=Float32)

            build_KEStore!(dh, mf, nnodes_loc, avg_mp_store)

            global u = zeros(ndofs(dh))

            log_entry = topopt_run(i, shear_stats, attempts, seed_used)
            push!(run_logs, log_entry)
        else
            println("RUN  - $(i) | shear cov μ_l=$(round(shear_stats.μ_l.cov, digits=4)), μ_t=$(round(shear_stats.μ_t.cov, digits=4)); attempts=$(attempts) -> aborted (threshold $(LOG_SHEAR_COV_THRESHOLD))")
            run_dir = WRITE_OUTPUT_FILES ? joinpath(save_path, "run_$(i)") : ""
            final_dir = WRITE_OUTPUT_FILES ? joinpath(run_dir, "final") : ""
            if WRITE_OUTPUT_FILES
                mkpath(final_dir)
            end

            x0 = ones(num_cells)
            density_history = Vector{DensityLogEntry}()
            log_density_metrics!(density_history, 0, x0)
            final_density_log = density_history[end]
            final_volume = sum(x0) / num_cells
            err_msg = "Shear COV threshold not met after $(attempts) attempts (μ_l cov=$(round(shear_stats.μ_l.cov, digits=4)), μ_t cov=$(round(shear_stats.μ_t.cov, digits=4)))"

            log_path = WRITE_OUTPUT_FILES ? write_run_log(run_dir;
                status=:invalid_material,
                shear_stats=shear_stats,
                resample_attempts=attempts,
                material_seed=seed_used,
                density_log=final_density_log,
                compliance=0.0,
                final_volume=final_volume,
                final_change=nothing,
                guard_threshold=LOG_SHEAR_COV_THRESHOLD,
                max_resamples=LOG_SHEAR_MAX_RESAMPLES,
                enforce_guard=LOG_ENFORCE_SHEAR_VARIATION,
                abort_on_guard=LOG_ABORT_ON_SHEAR_FAILURE,
                vtu_file=nothing,
                error=err_msg) : nothing

            push!(run_logs, (run_index=i,
                status=:invalid_material,
                error=err_msg,
                compliance=0.0,
                density_history=density_history,
                final_density_log=final_density_log,
                final_volume=final_volume,
                shear_stats=shear_stats,
                resample_attempts=attempts,
                material_seed=seed_used,
                output_dir=final_dir,
                vtu_file=nothing,
                log_file=log_path))
        end
    end

    return run_logs
end

function run_single_design(run_i::Int, coeffs_dict::Dict{Symbol,Vector{Float64}})
    # Generate fields with provided coeffs
    # Note: We use the same parameters as in sample_material_fields
    fields = KL_realization(mp, coords_elem;
        σs=Dict(:μ_l => 0.5,
            :μ_t => 0.5,
            :α => 0.5,
            :β => 0.5),
        Lc=0.01, N_modes=80, use_centroids=false,
        make_sparse=true, kernel=:exponential, mode=:lognormal,
        provided_coeffs=coeffs_dict)

    stats = shear_stats_from_fields(fields)

    mf = build_material_field(fields; use_centroids=false, eltype_out=Float32)

    # Update stiffness matrices
    build_KEStore!(dh, mf, nnodes_loc, avg_mp_store)

    # Reset global displacement vector
    global u = zeros(ndofs(dh))

    # Run optimization
    # We pass 0 for resample_attempts and nothing for seed since we provided coeffs
    log_entry = topopt_run(run_i, stats, 0, nothing)

    return log_entry
end

#run_logs = multiple_runs(3)
