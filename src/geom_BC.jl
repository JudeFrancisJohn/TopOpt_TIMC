using LinearAlgebra, Printf,SparseArrays,StaticArrays
using Plots,Dates,Statistics, DelimitedFiles
using Ferrite
using TickTock, Parameters, Random
using IterativeSolvers
using WriteVTK
using Distributions
using Arpack


include("../input/params_geom.jl")


corners = [
    Vec{2}((0.0, 0.0)),
    Vec{2}((lx, 0.0)),
    Vec{2}((lx, ly)),
    Vec{2}((0.0, ly)),
]

grid = generate_grid(Quadrilateral, (nelx, nely), corners)

#-------------------   Half-MBB Geometry ------------------------------------------#
addnodeset!(grid, "left_edge", x -> isapprox(x[1], 0.0; atol=1e-8));
addnodeset!(grid, "right_bottom_node", x -> isapprox(x[1], lx; atol=1e-8) && isapprox(x[2], 0.0; atol=1e-8));
addfaceset!(grid, "topmid_face", x -> (isapprox(x[2], ly; atol=1e-8) && abs(x[1]) <= 0.5))
#-------------------------------------------------------------------------------------#

dim = 2;
ip_g = Lagrange{dim, RefCube, 1}();
ip = Lagrange{dim, RefCube, 1}(); 
qpo = 2;
qr = QuadratureRule{dim, RefCube}(qpo);
cv = CellScalarValues(Float64,qr, ip, ip_g);
fqr = QuadratureRule{dim-1,RefCube}(qpo);
fv = FaceVectorValues(fqr, ip, ip_g);
nqp_post = 1;
qr_post = QuadratureRule{dim, RefCube}(nqp_post);
cv_post = CellScalarValues(Float64,qr_post, ip, ip_g);

dh = DofHandler(grid);
push!(dh, :u, 2, ip); # Displacement vector
close!(dh);
ch = ConstraintHandler(dh);

#-----------------         Dirichlet BC    ----------------------#


∂Ωl = getnodeset(dh.grid, "left_edge");
# Fix horizontal displacement on the entire left edge (roller in x)
add!(ch, Ferrite.Dirichlet(:u, ∂Ωl, (x,t) -> 0.0000, 1));

# Right edge: apply a vertical roller support (fix vertical DOF only)
∂Ωr = getnodeset(dh.grid, "right_bottom_node");
add!(ch, Ferrite.Dirichlet(:u, ∂Ωr, (x,t) -> 0.0000, 2));

"""∂Ωr = getnodeset(dh.grid, "right_edge");
dbcrv = Dirichlet(:u, ∂Ωr, (x,t) -> 0.2*t, 2); # Vertical Displacement
add!(ch, dbcrv);"""

close!(ch);
Ferrite.update!(ch, 0.0);  # Explicitly use Ferrite's update! to avoid ambiguity

#ΓN = getfaceset(grid, "right_face"); # Neumann Boundary
#--------------------------------------------------------------------#

# dof vector
u_d = Vector{Float64}(undef,ndofs(dh));
fill!(u_d,zero(eltype(u_d)));
u = Vector{Float64}(undef,ndofs(dh));
fill!(u,zero(eltype(u)));
u .= u_d;
