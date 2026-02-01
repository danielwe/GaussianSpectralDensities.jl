using CairoMakie: Axis3, Figure
# using GLMakie: Axis3, Figure

using GaussianSpectralDensities.Algorithms: RegularizerSchedule
using GaussianSpectralDensities.Ensembles: MatrixEnsemble
using GaussianSpectralDensities.GlobalData: GlobalProblem, GlobalSolution  #, zvals
using GaussianSpectralDensities.Problems: PTraceConstraint, ResolventEquation
using GaussianSpectralDensities.Regularizers: Regularizer, shift
using GaussianSpectralDensities.Solutions: issuccess
using GaussianSpectralDensities.Solvers: pointsolve
using LinearAlgebra: I
using Meshes:
    Ball,
    JarvisMarch,
    MaxLengthRefinement,
    Point,
    RegularDiscretization,
    SimpleMesh,
    coords,
    discretize,
    hull,
    refine,
    simplexify,
    viz!

disk = Ball(Point(0.0, 0.0), 1.5 * inv(cos(π / 6)))  # With this radius, the circle inscribed in the hexagon inscribed in the disk is the unit circle
hex = discretize(disk, RegularDiscretization(1, 6))
poly = hull(hex.vertices, JarvisMarch())  # GrahamScan is buggy and returns non-convex result
disc = discretize(poly)
mesh = simplexify(refine(disc, MaxLengthRefinement(0.2)))

ensemble = MatrixEnsemble(0I, I, I, I / 2)
equation = ResolventEquation(ensemble; shiftmin=-2)

constraint = PTraceConstraint(1; shift0=1, q0=0)
algorithm = RegularizerSchedule()

prob = GlobalProblem(equation, constraint, mesh)
sol = GlobalSolution(equation, mesh)

for i in 1:length(prob)
    sol[i] = pointsolve(prob[i], algorithm)
end

# FIXME: for failed vertices, inject NaN/nothing/missing in setindex!(::GlobalSolution, ...)
shiftmax = maximum(shift(sol))
prunedshift_vals = map(shift(sol), issuccess(sol)) do s, success
    return success ? max(s, -shiftmax / 2) : oftype(s, NaN)
end
shiftvertices = map(mesh.vertices, prunedshift_vals) do v, s
    c = coords(v)
    return Point(c.x.val, c.y.val, s)
end
shiftmesh = SimpleMesh(shiftvertices, mesh.topology)

fig = Figure()
ax = Axis3(fig[1, 1]; xlabel="Re(z)", ylabel="Im(z)", zlabel="shift")
viz!(
    ax,
    shiftmesh;
    color=prunedshift_vals,
    colorrange=(0.0, maximum(prunedshift_vals[isfinite.(prunedshift_vals)])),
)
fig
