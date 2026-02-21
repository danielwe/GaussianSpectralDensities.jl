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
    area_weighted_average, boundingbox, grad, mesh, updatemesh, wirtinger_derivatives

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
    grad(msh, s; label=Symbol(s, "_grad")) -> Mesh

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

function remove_faceviews(msh::Mesh)
    # Coloring a mesh plot with a vertex attribute when face views exist is broken. Probably
    # a Makie bug. For now, provide this convenience function to strip out the faceviews.
    # NOTE: filter applies the predicate to values, not key-value pairs, when applied to
    # NamedTuples (unlike when applied to Dicts). Surprising and undocumented, but OK.
    return Mesh(msh.faces; filter(v -> !(v isa FaceView), msh.vertex_attributes)...)
end

"""
    wirtinger_derivatives(mesh, s; label=Symbol(s, "_wirt") -> Mesh

Compute the Wirtinger derivatives of a scalar field on a 2D mesh over the complex plane

Returns a new mesh with an additional FaceView attribute `mesh.vertex_attributes[label]`
containing the Wirtinger derivatives. Each entry is an SVector of the form `(∂s/∂z, ∂s/∂z̄)`.
"""
function wirtinger_derivatives(
    mesh::Mesh{2,F}, s::Symbol; label=Symbol(s, "_wirt")
) where {F<:Real}
    grad_label = Symbol(s, "_grad")
    grad_msh = grad(mesh, s; label=grad_label)
    grd = grad_msh.vertex_attributes[grad_label]::FaceView
    # ∂f/∂z = (∂f/∂x - i∂f/∂y) / 2
    # ∂f/∂z̄ = (∂f/∂x + i∂f/∂y) / 2
    wirt_data = map(grd.data) do (gx, gy)::SVector{2}
        gx_2, igy_2 = gx / 2, im * gy / 2
        return SVector(gx_2 - igy_2, gx_2 + igy_2)
    end
    return updatemesh(grad_msh; label => per_face(wirt_data, grd.faces))
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

@testitem "grad and wirtinger_derivatives" begin
    using GaussianSpectralDensities.Meshing:
        area_weighted_average, grad, updatemesh, wirtinger_derivatives

    using GeometryBasics: FaceView, Mesh, Point2, TriangleFace
    using StaticArrays: SVector

    # Build a minimal mesh: unit square split into two triangles
    #   4--3
    #   |\ |
    #   | \|
    #   1--2
    vertices = Point2.([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    faces = TriangleFace.([(1, 2, 3), (1, 3, 4)])
    msh = Mesh(vertices, faces)

    @testset "grad with real-valued linear field" begin
        # f(x, y) = 2x + 3y — gradient should be exactly [2, 3] on every face
        vals = [2.0 * p[1] + 3.0 * p[2] for p in vertices]
        msh_f = updatemesh(msh; scalar=vals)

        msh_g = grad(msh_f, :scalar)
        gdata = msh_g.vertex_attributes[:scalar_grad]
        @test gdata isa FaceView
        @testset "face = $i" for (i, g) in pairs(gdata.data)
            @test g ≈ SVector(2.0, 3.0)
        end
    end

    @testset "grad with complex-valued linear field" begin
        vals = [(2.0 + 1.0im) * p[1] + (3.0 - 1.0im) * p[2] for p in vertices]
        msh_f = updatemesh(msh; scalar=vals)

        msh_g = grad(msh_f, :scalar)
        gdata = msh_g.vertex_attributes[:scalar_grad]
        @test gdata isa FaceView
        @testset "face = $i" for (i, g) in pairs(gdata.data)
            @test g ≈ SVector(2.0 + 1.0im, 3.0 - 1.0im)
        end
    end

    @testset "wirtinger_derivatives of anti-analytic function" begin
        # f(z) = z̄ = x - iy, exactly linear => ∂f/∂z = 0, ∂f/∂z̄ = 1 exactly
        vals = [p[1] - im * p[2] for p in vertices]
        msh_f = updatemesh(msh; scalar=vals)

        msh_w = wirtinger_derivatives(msh_f, :scalar)
        wdata = msh_w.vertex_attributes[:scalar_wirt]
        @test wdata isa FaceView
        @testset "face = $i" for (i, w) in pairs(wdata.data)
            @test w[1] ≈ 0  # ∂f/∂z
            @test w[2] ≈ 1  # ∂f/∂z̄
        end
    end

    @testset "wirtinger_derivatives of analytic function" begin
        # f(z) = z = x + iy => ∂f/∂z = 1, ∂f/∂z̄ = 0 exactly
        vals = [complex(p[1], p[2]) for p in vertices]
        msh_f = updatemesh(msh; scalar=vals)

        msh_w = wirtinger_derivatives(msh_f, :scalar)
        wdata = msh_w.vertex_attributes[:scalar_wirt]
        @test wdata isa FaceView
        @testset "face = $i" for (i, w) in pairs(wdata.data)
            @test w[1] ≈ 1  # ∂f/∂z
            @test w[2] ≈ 0  # ∂f/∂z̄
        end
    end

    @testset "area_weighted_average round-trip" begin
        # For a linear field the per-face gradient is constant, so area-weighted
        # averaging back to vertices should recover that same constant at every vertex.
        vals = [2.0 * p[1] + 3.0 * p[2] for p in vertices]
        msh_f = updatemesh(msh; scalar=vals)

        msh_g = grad(msh_f, :scalar)
        msh_a = area_weighted_average(msh_g, :scalar_grad)
        vg = msh_a.vertex_attributes[:vertex_scalar_grad]
        @testset "vertex = $i" for (i, g) in pairs(vg)
            @test g ≈ SVector(2.0, 3.0)
        end
    end
end
