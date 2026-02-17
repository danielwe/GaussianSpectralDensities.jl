module Densities

using LinearAlgebra: dot
using StaticArrays: SVector

using ..ArrayTools: RealOrComplex

public mesh_gradient_per_face, mesh_gradient_per_vertex, signed_face_area

"""
    mesh_gradient_per_face(values, vertices, faces) -> AbstractVector

Compute per-face gradients of a field over a complex mesh

Uses linear FEM basis functions, so gradient is constant over each triangular face. Returns
a vector where each entry is the gradient on the corresponding face in `faces`.
"""
function mesh_gradient_per_face(
    values::AbstractArray{T},
    vertices::AbstractArray{Complex{F}},
    faces::AbstractVector{<:SVector{3,<:Integer}},
) where {F,T<:RealOrComplex{F}}
    return map(faces) do face
        vals = face_vertex_vals(values, face)

        dzs = face_edge_dvals(vertices, face)
        dxs, dys = real.(dzs), imag.(dzs)
        signed_area2 = cross1(dxs, dys)

        # Important: pass the known real triple first first arg to avoid dot's conjugation
        dfdx = dot(dys, vals) / signed_area2
        dfdy = -dot(dxs, vals) / signed_area2
        return SVector(dfdx, dfdy)
    end
end

"""
    mesh_gradient_per_vertex(values, vertices, faces) -> AbstractVector

Compute per-vertex gradients of a field over a complex mesh

Gradient per face are computed as in `mesh_gradient_per_face`. Gradients per vertex are
obtained by area-weighted averaging.
"""
function mesh_gradient_per_vertex(
    values::AbstractArray{T},
    vertices::AbstractArray{Complex{F}},
    faces::AbstractVector{<:SVector{3,<:Integer}},
) where {F,T<:RealOrComplex{F}}
    face_grads = mesh_gradient_per_face(values, vertices, faces)
    return area_weighted_averages(face_grads, vertices, faces)
end

"""
     area_weighted_averages(face_values, vertices, faces) -> AbstractVector

Compute area-weighted averages of `face_values`

The value at each vertex is calculated as an average of the values at each of the adjacent
faces, weighted by the areas of those faces.
"""
function area_weighted_averages(
    face_values::AbstractVector{T},
    vertices::AbstractArray{Complex{F}},
    faces::AbstractVector{<:SVector{3,<:Integer}},
) where {F,T}
    zero_area = zero(F)^2
    zero_value = zero(T) * zero_area
    awvalues, aweights = fill(zero_value, size(vertices)), fill(zero_area, size(vertices))
    for (face, face_val) in zip(faces, face_values)
        area = abs(signed_face_area(vertices, face))
        for I in face
            awvalues[I] += area * face_val
            aweights[I] += area
        end
    end
    return awvalues ./ aweights
end

function signed_face_area(
    vertices::AbstractArray{T}, face::SVector{3,<:Integer}
) where {T<:Complex}
    dzs = face_edge_dvals(vertices, face)
    return cross1(real.(dzs), imag.(dzs)) / 2
end

cross1(a, b) = (a[2] * b[3]) - (a[3] * b[2])

function face_edge_dvals(vertex_vals::AbstractArray, face::SVector{3,<:Integer})
    face_vvals = face_vertex_vals(vertex_vals, face)
    return (
        face_vvals[2] - face_vvals[3],
        face_vvals[3] - face_vvals[1],
        face_vvals[1] - face_vvals[2],
    )
end

function face_vertex_vals(vertex_vals::AbstractArray, face::SVector{3,<:Integer})
    return ntuple(i -> vertex_vals[face[i]], Val(3))
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
