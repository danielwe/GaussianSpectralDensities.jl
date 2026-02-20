module Densities

using GeometryBasics: FaceView, Mesh, per_face

using ..Meshing: Meshing

public spectral_density

"""
    spectral_density(mesh, s=:greens; label=:density)

Compute the spectral density of a matrix ensemble from a mesh with resolvent parameters.

    density = |∂t/∂z̄| / π; here t is the ensemble's Green's function (normalized trace of
                           the ensemble-averaged resolvent).

Returns a new mesh with an additional FaceView attribute `mesh.vertex_attributes[label]`
containing the spectral density.
"""
function spectral_density(msh::Mesh{2}, s=:greens; label=:density)
    wirt_label = Symbol(s, "_wirt")
    wirt_msh = Meshing.wirtinger_derivatives(msh, s; label=wirt_label)
    wirt = wirt_msh.vertex_attributes[wirt_label]::FaceView
    density = real.(last.(wirt.data)) ./ π
    return Meshing.updatemesh(msh; label => per_face(density, wirt.faces))
end

end  # module Densities

#=
@testitem "element_gradients" begin
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.GlobalData:
        ElementGradient, GlobalSolution, element_gradients
    using GaussianSpectralDensities.Problems: ResolventEquation
    using LinearAlgebra: I
    using Meshes: Box, Point, RegularDiscretization, discretize, nelements, simplexify

    disc = discretize(Box(Point(0.0, 0.0), Point(1.0, 1.0)), RegularDiscretization(2, 2))
    mesh = simplexify(disc)
    eq = ResolventEquation(MatrixEnsemble(I, I, I, I / 2))

    sol = GlobalSolution(eq, mesh)
    shift_vals = 1:length(sol)
    sol.params.shift .= shift_vals
    sol.params.greens .= complex.(shift_vals, shift_vals .^ 2)

    grads_real = element_gradients(sol, :shift)
    @test length(grads_real) == nelements(mesh)
    @test grads_real[1] isa ElementGradient{Float64,Float64}
    @test all(g -> 0 <= real(g.centroid) <= 1 && 0 <= imag(g.centroid) <= 1, grads_real)

    grads_complex = element_gradients(sol, :greens)
    @test length(grads_complex) == nelements(mesh)
    @test grads_complex[1] isa ElementGradient{Float64,ComplexF64}
end
=#
