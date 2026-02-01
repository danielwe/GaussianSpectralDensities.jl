using CairoMakie: Axis3, Figure
# using GLMakie: Axis3, Figure

using GaussianSpectralDensities.Algorithms: RegularizerSchedule
using GaussianSpectralDensities.Ensembles: MatrixEnsemble
using GaussianSpectralDensities.GlobalData: GlobalProblem, GlobalSolution  #, zvals
using GaussianSpectralDensities.Problems: PTraceConstraint, ResolventEquation
using GaussianSpectralDensities.Regularizers: Regularizer
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

# FIXME: for failed vertices, inject NaN (or maybe nothing/missing) in
# setindex!(::GlobalSolution, ...)
shiftmax = maximum(sol.params.shift)
prunedshift = map(sol.params.shift, sol.success) do shift, success
    return success ? max(shift, -shiftmax / 2) : oftype(shift, NaN)
end
shiftvertices = map(mesh.vertices, prunedshift) do v, s
    c = coords(v)
    return Point(c.x.val, c.y.val, s)
end
shiftmesh = SimpleMesh(shiftvertices, mesh.topology)

fig = Figure()
ax = Axis3(fig[1, 1]; xlabel="Re(z)", ylabel="Im(z)", zlabel="shift")
viz!(
    ax,
    shiftmesh;
    color=prunedshift,
    colorrange=(0.0, maximum(prunedshift[isfinite.(prunedshift)])),
)
fig
