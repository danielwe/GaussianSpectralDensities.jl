module Meshing

using GeometryBasics: FaceView, Mesh, Point2, TriangleFace
using DistMesh: distmesh2d, huniform
using Statistics: mean
using StructArrays: components

using ..Solutions: issuccess
using ..Spreads: shift
using ..Surrogates: SurrogateModel, Surrogates, evaluate, evaluatebatch, zvals

public addresolventparams, boundingbox, mesh, remove_faceviews, updatemesh

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

function remove_faceviews(msh::Mesh)
    # Coloring a mesh plot with a vertex attribute when face views exist is broken. Probably
    # a Makie bug. For now, provide this convenience function to strip out the faceviews.
    # NOTE: filter applies the predicate to values, not key-value pairs, when applied to
    # NamedTuples (unlike when applied to Dicts). Surprising and undocumented, but OK.
    return Mesh(msh.faces; filter(v -> !(v isa FaceView), msh.vertex_attributes)...)
end

end  # module Meshing

@testitem "meshing" begin
    using GaussianSpectralDensities:
        DirectSolve, MatrixEnsemble, PTraceConstraint, ResolventEquation
    using GaussianSpectralDensities.Meshing: mesh
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

@testitem "remove_faceviews" begin
    using GaussianSpectralDensities.Meshing: remove_faceviews, updatemesh

    using GeometryBasics: FaceView, Mesh, Point2, TriangleFace, per_face

    vertices = Point2.([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    faces = TriangleFace.([(1, 2, 3), (1, 3, 4)])
    msh = Mesh(vertices, faces)

    vertexvals = [2.0 * p[1] + 3.0 * p[2] for p in vertices]
    msh_f = updatemesh(msh; vertexvals=vertexvals)

    facevals = [1.0, 2.0]
    msh_g = updatemesh(msh_f; facevals=per_face(facevals, msh))

    @test msh_g.vertex_attributes[:facevals] isa FaceView

    # remove_faceviews should strip FaceViews but keep non-FaceView attrs
    msh_clean = remove_faceviews(msh_g)
    @test !haskey(msh_clean.vertex_attributes, :facevals)
    @test haskey(msh_clean.vertex_attributes, :vertexvals)
end
