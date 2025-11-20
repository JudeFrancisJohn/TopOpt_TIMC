function OC(x::Vector{Float64}, volfrac::Float64, dc::Vector{Float64})
    l1 = 0.0
    l2 = 1e9
    move = 0.2
    eta = 0.5
    xnew = similar(x)
    while (l2 - l1) / (l1 + l2) > 1e-3
        lmid = 0.5 * (l2 + l1)
        xnew .= clamp.(x .* sqrt.(-dc ./ lmid), x .- move, x .+ move)
        xnew .= clamp.(xnew, 0.001, 1.0)
        if mean(xnew) - volfrac > 0
            l1 = lmid
        else
            l2 = lmid
        end
    end
    return xnew
end

# --- Function: Sensitivity filter ---
function check(nelx::Integer, nely::Integer,rmin::Float64, x::Vector{Float64}, dc::Vector{Float64})
    num_cells = length(x)
    dcn = zeros(num_cells)
    for i = 1:nelx
        for j = 1:nely
            cell_index = (j-1) * nelx + i
            sum_fac = 0.0
            for k = max(i - Int(floor(rmin)), 1):min(i + Int(floor(rmin)), nelx)
                for l = max(j - Int(floor(rmin)), 1):min(j + Int(floor(rmin)), nely)
                    fac = max(0.0, rmin - sqrt((i - k)^2 + (j - l)^2))
                    sum_fac += fac
                    cell_j = (l-1) * nelx + k
                    dcn[cell_index] += fac * x[cell_j] * dc[cell_j]
                end
            end
            dcn[cell_index] /= (x[cell_index] * sum_fac + 1e-8)
        end
    end
    return dcn
end