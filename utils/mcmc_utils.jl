function evaluate_badness(log_entry)
    # If the run failed, return a score of 0.0 (or very low).
    if log_entry.status != :success
        return 0.0
    end

    props = if hasproperty(log_entry, :final_density_log)
        getfield(log_entry, :final_density_log)
    elseif isa(log_entry, AbstractDict) && haskey(log_entry, :final_density_log)
        log_entry[:final_density_log]
    else
        return 0.0
    end

    bin_props = if hasproperty(props, :bin_proportions)
        getfield(props, :bin_proportions)
    elseif isa(props, AbstractDict) && haskey(props, :bin_proportions)
        props[:bin_proportions]
    else
        return 0.0
    end

    # Focus on intermediary density (bin 2)
    mid_prop = bin_props[2]
    mid_density_term = DENSITY_WEIGHT * mid_prop

    # Compliance term (optional, very small weight)
    comp = get_compliance(log_entry)
    compliance_term = if comp === nothing
        0.0
    else
        COMPLIANCE_WEIGHT * comp
    end

    # Total badness: prioritize intermediary density
    return mid_density_term + compliance_term
end

function write_iteration_record(io, iteration; accepted::Bool, is_burnin::Bool, current_score::Float64, proposal_score::Float64, acceptance_prob::Float64, running_acceptance::Float64, log_entry, coeffs::Dict{Symbol,Vector{Float64}})
    println(io, "Iteration=$(iteration)")
    println(io, "Accepted=$(accepted)")
    println(io, "Is_Burnin=$(is_burnin)")
    println(io, "Score=$(round(current_score, digits=6))")
    println(io, "Proposal_Score=$(round(proposal_score, digits=6))")
    println(io, "Acceptance_Prob=$(round(acceptance_prob, digits=6))")
    println(io, "Running_Acceptance=$(round(running_acceptance, digits=6))")
    status = hasproperty(log_entry, :status) ? getfield(log_entry, :status) : (isa(log_entry, AbstractDict) && haskey(log_entry, :status) ? log_entry[:status] : "unknown")
    println(io, "Status=$(status)")
    compliance = get_compliance(log_entry)
    compliance_str = compliance === nothing ? "N/A" : string(round(compliance, digits=6))
    println(io, "Compliance=$(compliance_str)")
    density_info = extract_density_log(log_entry)
    bin_counts = isnothing(density_info) ? "N/A" : string(density_info.bin_counts)
    bin_props = isnothing(density_info) ? "N/A" : string(density_info.bin_proportions)
    poor_design_str = isnothing(density_info) ? "N/A" : string(density_info.poor_design)
    println(io, "Density_Bin_Counts=$(bin_counts)")
    println(io, "Density_Bin_Proportions=$(bin_props)")
    println(io, "Poor_Design=$(poor_design_str)")
    println(io, "Coefficients:")
    for (sym, vals) in sort!(collect(coeffs); by=first)
        coeffs_str = join(string.(round.(vals, digits=6)), ", ")
        println(io, "  $(sym)=[", coeffs_str, "]")
    end
    println(io)
end