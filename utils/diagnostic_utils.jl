module DiagnosticUtils

using Plots
using Statistics

# Initialize global stores for diagnostics
const trace_data = Dict{Symbol, Vector{Float64}}()
const acceptance_rates = Float64[]
const iteration_numbers = Int[]
const ess_history = Dict{Symbol, Vector{Float64}}()

# Function to initialize diagnostics
function initialize_diagnostics!(params::Vector{Symbol})
    empty!(trace_data)
    empty!(ess_history)
    empty!(acceptance_rates)
    empty!(iteration_numbers)
    for param in params
        trace_data[param] = Float64[]
        ess_history[param] = Float64[]
    end
end

# Function to update diagnostics
function update_diagnostics!(current_params::Dict{Symbol, Float64}, acceptance_rate::Float64, iteration::Int)
    for (param, value) in current_params
        values = trace_data[param]
        push!(values, value)
        maxlag = clamp(div(length(values), 2), 1, 200)
        push!(ess_history[param], effective_sample_size(values; maxlag=maxlag))
    end
    push!(acceptance_rates, acceptance_rate)
    push!(iteration_numbers, iteration)
end

# Function to plot diagnostics
function plot_diagnostics!()
    # Plot parameter traces
    for (param, values) in trace_data
        xvals = iteration_numbers[1:length(values)]
        fig = plot(xvals, values; label=string(param), xlabel="Iteration", ylabel="Value", title="Trace for $(param)")
        display(fig)
    end

    # Plot acceptance rate
    fig_accept = plot(iteration_numbers, acceptance_rates; xlabel="Iteration", ylabel="Acceptance Rate", title="Acceptance Rate")
    display(fig_accept)

    # Plot effective sample size estimate
    for (param, values) in ess_history
        xvals = iteration_numbers[1:length(values)]
        fig = plot(xvals, values; label=string(param), xlabel="Iteration", ylabel="ESS", title="ESS for $(param)")
        display(fig)
    end
end

# Function to monitor diagnostics during runtime
function monitor_diagnostics!(current_params::Dict{Symbol, Float64}, acceptance_rate::Float64, iteration::Int, plot_interval::Int)
    update_diagnostics!(current_params, acceptance_rate, iteration)
    if iteration % plot_interval == 0
        plot_diagnostics!()
    end
end

# Fallback autocorrelation computation if StatsBase.autocor is unavailable
function _autocorrelation_fallback(series::AbstractVector{<:Real}, maxlag::Int)
    n = length(series)
    μ = mean(series)
    σ2 = var(series; mean=μ, corrected=false)
    σ2 == 0 && return ones(Float64, maxlag + 1)
    ac = zeros(Float64, maxlag + 1)
    for lag in 0:maxlag
        s = 0.0
        for i in 1:(n - lag)
            s += (series[i] - μ) * (series[i + lag] - μ)
        end
        ac[lag + 1] = s / ((n - lag) * σ2)
    end
    return ac
end

function autocorrelation(series::AbstractVector{<:Real}; maxlag::Int = min(length(series) - 1, 200))
    maxlag = max(0, min(maxlag, length(series) - 1))
    return _autocorrelation_fallback(series, maxlag)
end

function effective_sample_size(series::AbstractVector{<:Real}; maxlag::Int = min(length(series) - 1, 200))
    n = length(series)
    n < 2 && return float(n)
    ac = autocorrelation(series; maxlag=maxlag)
    # Use the initial monotone sequence estimator heuristic (stop when pair sum <= 0)
    τ = 1.0
    for k in 2:length(ac)
        ρ = ac[k]
        if ρ <= 0
            break
        end
        τ += 2ρ
    end
    τ = max(τ, 1.0)
    return n / τ
end

function summarize_autocorrelation(series_dict::Dict{Symbol, Vector{Float64}}; maxlag::Int = 100)
    ac_summary = Dict{Symbol, Vector{Float64}}()
    for (param, values) in series_dict
        ac_summary[param] = autocorrelation(values; maxlag=maxlag)
    end
    return ac_summary
end

function summarize_ess(series_dict::Dict{Symbol, Vector{Float64}}; maxlag::Int = 100)
    ess_summary = Dict{Symbol, Float64}()
    for (param, values) in series_dict
        ess_summary[param] = effective_sample_size(values; maxlag=maxlag)
    end
    return ess_summary
end

end # module