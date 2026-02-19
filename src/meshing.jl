module Meshing

using GeometryBasics: FaceView, Mesh, Point2, TriangleFace, per_face
using DistMesh: distmesh2d, huniform
using LinearAlgebra: dot
using StaticArrays: SVector, StaticVector
using Statistics: mean
using StructArrays: components

using ..Solutions: issuccess
using ..Spreads: shift
using ..Surrogates: SurrogateModel, Surrogates, evaluate, evaluatebatch, zvals

public addresolventparams,
    area_weighted_average, boundingbox, grad, mesh, signed_face_area, updatemesh

function mesh(model::SurrogateModel; hcoeff::Real=1, shiftoffset::Real=0, params=false)
    h = hcoeff * mean(model.rbf.nn_dists)
    bbox = boundingbox(model.rbf.x)
    dmsh = distmesh2d(huniform, h, bbox) do p
        sol = evaluate(model, complex(p...); refine=false)
        s = shift(sol)
        return issuccess(sol) ? -(s + shiftoffset) : oftype(s, Inf)
    end
    msh = Mesh(map(Point2, dmsh.p), map(TriangleFace, dmsh.t))
    return params ? addresolventparams(msh, model) : msh
end

function mesh(eq, constraint, alg, args...; kwargs...)
    return mesh(SurrogateModel(eq, constraint, alg, args...); kwargs...)
end

function boundingbox(ps::AbstractVector)
    ffirst, flast = (eltype(ps) <: Complex) ? (real, imag) : (first, last)
    xmin, xmax = extrema(ffirst, ps)
    ymin, ymax = extrema(flast, ps)
    return ((xmin, ymin), (xmax, ymax))
end

Surrogates.zvals(msh::Mesh) = map(splat(complex), msh.position)

function addresolventparams(msh::Mesh, model::SurrogateModel)
    sols = evaluatebatch(model, zvals(msh); refine=false)
    return updatemesh(msh; components(sols.params)...)
end

function updatemesh(msh; attrs...)
    # GeometryBasics.mesh is broken, downconverts to Float32 unprompted. Implement our own.
    return Mesh(msh.faces; msh.vertex_attributes..., attrs...)
end

"""
    grad(msh, s) -> Mesh

Compute per-face gradients of a field over a 2D mesh

Uses linear FEM basis functions, so gradient is constant over each triangular face.

Returns a new mesh with a `FaceView` attribute containing the gradient.
"""
function grad(msh::Mesh, s::Symbol; label=Symbol(s, "_grad"))
    g = grad(msh.vertex_attributes[s], msh.position, msh.faces)
    return updatemesh(msh; label => per_face(g, msh))
end

function grad(
    values::AbstractArray{<:Any,N},
    vertices::AbstractArray{<:StaticVector{2,<:Real},N},
    faces::AbstractVector{<:StaticVector{3,<:Integer}},
) where {N}
    return map(faces) do face
        vals = face_vertex_vals(values, face)

        dps = face_edge_dvals(vertices, face)
        dxs, dys = first.(dps), last.(dps)
        signed_area2 = cross1(dxs, dys)

        # Important: pass the coordinate triple, always real, as first arg to avoid dot's conjugation
        dfdx = dot(dys, vals) / signed_area2
        dfdy = -dot(dxs, vals) / signed_area2
        return SVector(dfdx, dfdy)
    end
end

function signed_face_area(
    vertices::AbstractArray{<:StaticVector{2,<:Real}}, face::StaticVector{3,<:Integer}
)
    dps = face_edge_dvals(vertices, face)
    return cross1(first.(dps), last.(dps)) / 2
end

cross1(a, b) = (a[2] * b[3]) - (a[3] * b[2])

function face_edge_dvals(vertex_vals::AbstractArray, face::StaticVector{3,<:Integer})
    face_vvals = face_vertex_vals(vertex_vals, face)
    return (
        face_vvals[2] - face_vvals[3],
        face_vvals[3] - face_vvals[1],
        face_vvals[1] - face_vvals[2],
    )
end

function face_vertex_vals(vertex_vals::AbstractArray, face::StaticVector{3,<:Integer})
    return ntuple(i -> vertex_vals[face[i]], Val(3))
end

"""
    area_weighted_average(msh, s; label=Symbol("vertex_", s)) -> Mesh

Map the values of the face attribute `msh.vertex_attributes[s]::FaceView` onto vertices

Each vertex value is calculated as an average of the value at each of the adjacent faces,
weighted by the face areas.

Returns a new mesh instance with an additional attribute `new_msh.vertex_attributes[label]`.
"""
function area_weighted_average(msh::Mesh, s::Symbol; label=Symbol("vertex_", s))
    face_values = msh.vertex_attributes[s]::FaceView
    vertex_values = area_weighted_average(face_values.data, msh.position, msh.faces)
    return updatemesh(msh; label => vertex_values)
end

function area_weighted_average(
    face_values::AbstractVector{T},
    vertices::AbstractArray{<:StaticVector{2,F}},
    faces::AbstractVector{<:StaticVector{3,<:Integer}},
) where {F<:Real,T}
    zero_area = zero(F)^2
    zero_value = zero(T) * zero_area
    aw_values, aweights = fill(zero_value, size(vertices)), fill(zero_area, size(vertices))
    for (face, face_val) in zip(faces, face_values)
        area = abs(signed_face_area(vertices, face))
        for I in face
            aw_values[I] += area * face_val
            aweights[I] += area
        end
    end
    return aw_values ./ aweights
end

end  # module Meshing

@testitem "meshing" begin
    using GaussianSpectralDensities.Algorithms: DirectSolve
    using GaussianSpectralDensities.Ensembles: MatrixEnsemble
    using GaussianSpectralDensities.Meshing: mesh
    using GaussianSpectralDensities.Problems: PTraceConstraint, ResolventEquation
    using GaussianSpectralDensities.Surrogates: SurrogateModel, zvals

    using GeometryBasics: Mesh
    using LinearAlgebra: I

    ensemble = MatrixEnsemble(0.0I, I, I, 0.5I)
    eq = ResolventEquation(ensemble)
    constraint = PTraceConstraint(1.0; gap0=1.0, shape0=complex(0.1))
    alg = DirectSolve()

    coarse = mesh(eq, constraint, alg, 16; hcoeff=1 // 2, shiftoffset=1 // 6)
    @test coarse isa Mesh{2,Float64}
    @test !isempty(coarse.position)
    @test !isempty(coarse.faces)

    coarse_model = SurrogateModel(eq, constraint, alg, zvals(coarse))
    fine = mesh(coarse_model; hcoeff=1 // 2)
    @test fine isa Mesh{2,Float64}
    @test !isempty(fine.position)
    @test length(fine.position) > length(coarse.position)

    nv = length(fine.position)
    @test all(f -> all(i -> 1 <= i <= nv, f), fine.faces)
end
