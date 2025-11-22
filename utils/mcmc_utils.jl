
# --- Helper Functions ---

"""
Evaluate the 'badness' of a design.
Higher score = 'worse' design (which is what we want to find).
"""
function get_compliance(log_entry)
    for name in (:compliance, :final_compliance, :objective, :final_objective)
        if hasproperty(log_entry, name)
            return getfield(log_entry, name)
        elseif isa(log_entry, AbstractDict) && haskey(log_entry, name)
            return log_entry[name]
        end
    end
    return nothing
end
