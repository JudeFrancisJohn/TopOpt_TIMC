struct topoptmetrics
    run_id::Int
    kle_seed::Int
    grayness::Float64
    volume_error::Float64
    compliance::Float64
    variance::Float64
    λmin::Float64
    label::Symbol      # :good, :smeared, :failed, etc.
end


include("stochastic_modified_v2.jl")